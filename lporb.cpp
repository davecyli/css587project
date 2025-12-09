/*
* David Li, Ben Schipunov, Kris Yu
 * CSS 587 - Final Project: LP-SIFT
 *
 * lporb.cpp
 * LP-SIFT implementation based on:
 * Hao Li et al., "Local-peak scale-invariant feature transform for fast and random image stitching"
 * (arXiv:2405.08578v2).
 *
 * The detector follows the paper's key idea:
 *  - Section 2.1 Image Preprocessing
 *      Add a tiny linear background (alpha) to avoid flat regions with identical intensities.
 *  - Section 2.2 Feature Point Detection
 *      Partition the image into interrogation windows of multiple sizes (L).
 *      Collect both the local maximum and minimum within each window as keypoints (multiscale peaks).
 *  - Section 2.3 - Feature Point Description
 *      [Experimental] Use ORB descriptors around those peak points.
 */

#include "lporb.h"

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>

cv::Ptr<LPORB> LPORB::create(const std::vector<int>& windowSizes,
                             const float linearNoiseAlpha) {
    return cv::makePtr<LPORB>(windowSizes, linearNoiseAlpha);
}

LPORB::LPORB(const std::vector<int>& windowSizes,
             const float linearNoiseAlpha)
    : descriptor_(cv::ORB::create()),
      windowSizes_(windowSizes),
      linearNoiseAlpha_(linearNoiseAlpha) {}

cv::String LPORB::getDefaultName() const {
    return "Feature2D.LPORB";
}

// Section 2.1: Image Preprocessing
// Adds alpha * (y * cols + x) to each pixel to break flat plateaus deterministically.
// Minima and maxima are biased top to bottom if a window is perfectly flat.
void LPORB::addLinearRamp(cv::Mat& image) const {
    // Input checks
    if (linearNoiseAlpha_ <= 0.0f || image.empty()) return;

    // Pre-compute ramp and add to image
    cv::Mat ramp(image.rows, image.cols, CV_32F);
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
                                  std::vector<cv::KeyPoint>& out) {
    if (x < 0 || y < 0 || x >= cols || y >= rows || windowSize <= 0) return false;

    const auto size = static_cast<float>(windowSize);
    cv::KeyPoint kp(cv::Point2f(static_cast<float>(x), static_cast<float>(y)), size);
    kp.response = response;
    kp.angle = -1.0f; // let ORB assign orientation during compute()
    kp.octave = octaveIndex;
    kp.class_id = windowSize; // store interrogation window size for analysis
    out.push_back(kp);
    return true;
}

/// Section 2.2 Feature Point Detection
void LPORB::detect(cv::InputArray image,
                   std::vector<cv::KeyPoint>& keypoints,
                   cv::InputArray mask) {
    CV_UNUSED(mask); // Mask input is kept for API compatibility. Not implemented.
    keypoints.clear();

    // Early exit if image is empty
    if (image.empty()) return;

    const cv::Mat src = image.getMat();

    cv::Mat gray;
    if (src.channels() > 1) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
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
                cv::Rect roi(x, y, L, L);

                cv::Mat tile = gray(roi);
                double minVal = 0.0, maxVal = 0.0;
                cv::Point minLoc, maxLoc;

                cv::minMaxLoc(tile, &minVal, &maxVal, &minLoc, &maxLoc);

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
}

/// Section 2.3 Feature Point Description
void LPORB::compute(cv::InputArray image,
                    std::vector<cv::KeyPoint>& keypoints,
                    cv::OutputArray descriptors) {
    if (keypoints.empty()) {
        descriptors.release();
        return;
    }

    const cv::Mat src = image.getMat();
    if (src.empty() || descriptor_.empty()) {
        descriptors.release();
        return;
    }

    cv::Mat gray;
    if (src.channels() > 1) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src;
    }

    if (gray.type() != CV_8U) {
        gray.convertTo(gray, CV_8U);
    }

    descriptor_->compute(gray, keypoints, descriptors); // Experimental ORB descriptor on LP-ORB keypoints
}

void LPORB::detectAndCompute(cv::InputArray image,
                             cv::InputArray mask,
                             std::vector<cv::KeyPoint>& keypoints,
                             cv::OutputArray descriptors,
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
