/*
 * LP-SIFT implementation based on:
 * Hao Li et al., "Local-peak scale-invariant feature transform for fast and random image stitching"
 * (arXiv:2405.08578v2).
 *
 * The detector follows the paper's key idea:
 *  - Add a tiny linear background (alpha) to avoid flat regions with identical intensities.
 *  - Partition the image into interrogation windows of multiple sizes (L).
 *  - Collect both the local maximum and minimum within each window as keypoints (multi-scale peaks).
 *  - Use SIFT descriptors around those peak points.
 */

#include "lpsift.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>
#include <unordered_set>

using namespace cv;

namespace {
// Hash key for deduplication: include scale to allow the same location at different L values.
inline uint64_t makeKey(int x, int y, int L) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(L)) << 40) ^
           (static_cast<uint64_t>(static_cast<uint32_t>(y)) << 20) ^
           static_cast<uint32_t>(x);
}
} // namespace

Ptr<LPSIFT> LPSIFT::create(const std::vector<int>& windowSizes,
                           const float linearNoiseAlpha,
                           const float beta0,
                           const int subregionGrid) {
    return makePtr<LPSIFT>(windowSizes, linearNoiseAlpha, beta0, subregionGrid);
}

LPSIFT::LPSIFT(const std::vector<int>& windowSizes,
               const float linearNoiseAlpha,
               const float beta0,
               const int subregionGrid)
    : descriptor_(SIFT::create()),
      windowSizes_(windowSizes),
      linearNoiseAlpha_(linearNoiseAlpha),
      beta0_(beta0),
      subregionGrid_(std::max(1, subregionGrid)) {}

String LPSIFT::getDefaultName() const {
    return "Feature2D.LPSIFT";
}

float LPSIFT::computePatchSize(const int windowSize) const {
    if (windowSize <= 0) return 0.0f;

    int Lmax = windowSize;
    if (!windowSizes_.empty()) {
        Lmax = *std::max_element(windowSizes_.begin(), windowSizes_.end());
    }

    const float beta = beta0_ * std::pow(static_cast<float>(windowSize) / static_cast<float>(Lmax), -0.5f);
    const float rawSize = beta * static_cast<float>(windowSize) * static_cast<float>(subregionGrid_);

    // Clamp to avoid degenerate patches and to stay inside the window.
    return std::clamp(rawSize, 4.0f, static_cast<float>(windowSize));
}

void LPSIFT::addLinearRamp(Mat& image) const {
    if (linearNoiseAlpha_ <= 0.0f || image.empty()) return;
    CV_Assert(image.type() == CV_32F || image.type() == CV_32FC1);

    const int cols = image.cols;
    for (int y = 0; y < image.rows; ++y) {
        float* row = image.ptr<float>(y);
        const float base = linearNoiseAlpha_ * static_cast<float>(y * cols);
        for (int x = 0; x < cols; ++x) {
            row[x] += base + linearNoiseAlpha_ * static_cast<float>(x);
        }
    }
}

void LPSIFT::detect(InputArray image,
                    std::vector<KeyPoint>& keypoints,
                    InputArray mask) {
    keypoints.clear();

    Mat src = image.getMat();
    if (src.empty()) return;

    Mat gray;
    if (src.channels() > 1) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    gray.convertTo(gray, CV_32F);
    addLinearRamp(gray);

    Mat mask8u;
    if (!mask.empty()) {
        mask.getMat().convertTo(mask8u, CV_8U);
        if (mask8u.size() != gray.size()) {
            // Mismatched mask size; ignore the mask to avoid indexing errors.
            mask8u.release();
        }
    }

    std::vector<int> defaultWindows{64};
    const std::vector<int>& windows = windowSizes_.empty() ? defaultWindows : windowSizes_;

    const int rows = gray.rows;
    const int cols = gray.cols;

    std::unordered_set<uint64_t> seen;

    auto tryAddKeypoint = [&](int x, int y, int L, float response) {
        if (x < 0 || y < 0 || x >= cols || y >= rows) return;
        uint64_t key = makeKey(x, y, L);
        if (!seen.insert(key).second) return;

        KeyPoint kp(Point2f(static_cast<float>(x), static_cast<float>(y)), computePatchSize(L));
        kp.response = response;
        kp.angle = -1.0f;      // let SIFT assign orientation during compute()
        kp.octave = 0;         // keep octave at 0 to satisfy SIFT's expectations
        kp.class_id = L;       // store interrogation window size for analysis
        keypoints.push_back(kp);
    };

    for (int L : windows) {
        if (L <= 0) continue;

        for (int y = 0; y < rows; y += L) {
            const int h = std::min(L, rows - y);
            for (int x = 0; x < cols; x += L) {
                const int w = std::min(L, cols - x);
                Rect roi(x, y, w, h);

                Mat tile = gray(roi);
                double minVal = 0.0, maxVal = 0.0;
                Point minLoc, maxLoc;

                if (!mask8u.empty()) {
                    Mat maskROI = mask8u(roi);
                    if (countNonZero(maskROI) == 0) continue;
                    minMaxLoc(tile, &minVal, &maxVal, &minLoc, &maxLoc, maskROI);
                } else {
                    minMaxLoc(tile, &minVal, &maxVal, &minLoc, &maxLoc);
                }

                const int gxMax = x + maxLoc.x;
                const int gyMax = y + maxLoc.y;
                const int gxMin = x + minLoc.x;
                const int gyMin = y + minLoc.y;

                // Add the maximum peak
                tryAddKeypoint(gxMax, gyMax, L, static_cast<float>(maxVal - minVal));

                // Add the minimum peak if it is distinct
                if (gxMin != gxMax || gyMin != gyMax) {
                    tryAddKeypoint(gxMin, gyMin, L, static_cast<float>(maxVal - minVal));
                }
            }
        }
    }

    // Keep the strongest peaks first (helps downstream limitKeypoints calls).
    std::sort(keypoints.begin(), keypoints.end(),
              [](const KeyPoint& a, const KeyPoint& b) {
                  return a.response > b.response;
              });
}

void LPSIFT::compute(InputArray image,
                     std::vector<KeyPoint>& keypoints,
                     OutputArray descriptors) {
    if (keypoints.empty()) {
        descriptors.release();
        return;
    }

    Mat src = image.getMat();
    if (src.empty() || descriptor_.empty()) {
        descriptors.release();
        return;
    }

    Mat gray;
    if (src.channels() > 1) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src;
    }

    if (gray.type() != CV_8U) {
        gray.convertTo(gray, CV_8U);
    }

    descriptor_->compute(gray, keypoints, descriptors); // Use SIFT descriptor implementation
}

void LPSIFT::detectAndCompute(InputArray image,
                              InputArray mask,
                              std::vector<KeyPoint>& keypoints,
                              OutputArray descriptors,
                              const bool useProvidedKeypoints) {
    if (!useProvidedKeypoints) {
        detect(image, keypoints, mask);
    }

    if (keypoints.empty()) {
        descriptors.release();
        return;
    }

    compute(image, keypoints, descriptors);
}
