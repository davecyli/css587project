/*
 * LP-ORB implementation based on but with the ORB descriptor:
 * Hao Li et al., "Local-peak scale-invariant feature transform for fast and random image stitching"
 * (arXiv:2405.08578v2).
 *
 * The detector follows the paper's key idea:
 *  - Add a tiny linear background (alpha) to avoid flat regions with identical intensities.
 *  - Partition the image into interrogation windows of multiple sizes (L).
 *  - Collect both the local maximum and minimum within each window as keypoints (multi-scale peaks).
 *  - Use ORB descriptors around those peak points.
 */

#include "lporb.h"

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

using namespace cv;

Ptr<LPORB> LPORB::create(const std::vector<int>& windowSizes,
                           const float linearNoiseAlpha) {
    return makePtr<LPORB>(windowSizes, linearNoiseAlpha);
}

LPORB::LPORB(const std::vector<int>& windowSizes,
               const float linearNoiseAlpha)
    : descriptor_(ORB::create()),
      windowSizes_(windowSizes),
      linearNoiseAlpha_(linearNoiseAlpha) {}

String LPORB::getDefaultName() const {
    return "Feature2D.LPORB";
}

// Step 1: Image Preprocessing
// Adds alpha * (y * cols + x) to each pixel to break flat plateaus deterministically.
// Based on 𝑀𝑛,𝑘(𝑖,𝑗)=𝑀𝑘(𝑖,𝑗)+[(𝑖−1)∗𝑛𝑐𝑘 +𝑗]∗𝛼 where 𝛼 ≪ 1 is the linearNoiseAlpha
// Precondition: image.type() == CV_32F
void LPORB::addLinearRamp(Mat& image) const {
    // Input checks
    if (linearNoiseAlpha_ <= 0.0f || image.empty()) return;

    // Pre-compute ramp and add to image
    Mat ramp(image.rows, image.cols, CV_32F);
    auto* data = ramp.ptr<float>();
    std::iota(data, data + ramp.total(), 0.0f); // 0,1,2,... in raster order
    ramp *= linearNoiseAlpha_;
    image += ramp;
}

bool LPORB::addKeypointCandidate(const int x,
                                  const int y,
                                  const int windowSize,
                                  const int octaveIndex,
                                  const float response,
                                  const int cols,
                                  const int rows,
                                  std::vector<KeyPoint>& out) {
    if (x < 0 || y < 0 || x >= cols || y >= rows || windowSize <= 0) return false;

    const auto size = static_cast<float>(windowSize);
    KeyPoint kp(Point2f(static_cast<float>(x), static_cast<float>(y)), size);
    kp.response = response;
    kp.angle = -1.0f; // let SIFT assign orientation during compute()
    kp.octave = octaveIndex;
    kp.class_id = windowSize; // store interrogation window size for analysis
    out.push_back(kp);
    return true;
}

// Step 2: Feature Point Detection
// This function integrates addLinearRamp (Step 1) in the call.
// Precondition: windowSizes_ is not empty and that the value is greater than 1
void LPORB::detect(InputArray image,
                    std::vector<KeyPoint>& keypoints,
                    InputArray mask) {
    CV_UNUSED(mask); // Mask input is kept for API compatibility. Not implemented.
    keypoints.clear();

    // Early exit if image is empty
    if (image.empty()) return;

    const Mat src = image.getMat();

    Mat gray;
    if (src.channels() > 1) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    gray.convertTo(gray, CV_32F);
    addLinearRamp(gray);

    const int rows = gray.rows;
    const int cols = gray.cols;

    for (size_t idx = 0; idx < windowSizes_.size(); ++idx) {
        const int L = windowSizes_[idx];
        for (int y = 0; y + L <= rows; y += L) {
            for (int x = 0; x + L <= cols; x += L) {
                Rect roi(x, y, L, L);

                Mat tile = gray(roi);
                double minVal = 0.0, maxVal = 0.0;
                Point minLoc, maxLoc;

                minMaxLoc(tile, &minVal, &maxVal, &minLoc, &maxLoc);

                const int gxMax = x + maxLoc.x;
                const int gyMax = y + maxLoc.y;
                const int gxMin = x + minLoc.x;
                const int gyMin = y + minLoc.y;
                const auto response = static_cast<float>(maxVal - minVal);

                addKeypointCandidate(gxMax, gyMax, L, static_cast<int>(idx), response, cols, rows, keypoints);
                addKeypointCandidate(gxMin, gyMin, L, static_cast<int>(idx), response, cols, rows, keypoints);
            }
        }
    }

    // // Keep the strongest peaks first (helps downstream limitKeypoints calls).
    // std::sort(keypoints.begin(), keypoints.end(),
    //           [](const KeyPoint& a, const KeyPoint& b) {
    //               return a.response > b.response;
    //           });
}

void LPORB::compute(InputArray image,
                     std::vector<KeyPoint>& keypoints,
                     OutputArray descriptors) {
    if (keypoints.empty()) {
        descriptors.release();
        return;
    }

    const Mat src = image.getMat();
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

    // ORB descriptor on LP-ORB keypoints
    descriptor_->compute(gray, keypoints, descriptors);
}

void LPORB::detectAndCompute(InputArray image,
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
