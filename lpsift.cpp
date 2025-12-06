/*
 * LP-SIFT implementation based on:
 * Hao Li et al., "Local-peak scale-invariant feature transform for fast and random image stitching"
 * (arXiv:2405.08578v2).
 *
 * This implementation follows the original MATLAB code:
 *  - Partition image into interrogation windows of multiple sizes (L)
 *  - Add linear noise to break flat regions
 *  - Collect local max/min within each window as keypoints
 *  - Filter keypoints using 3x3 neighborhood uniqueness check
 *  - Compute custom 64-dimensional descriptors using dx/dy gradients
 */

#include "lpsift.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include <algorithm>
#include <numeric>
#include <cmath>

using namespace cv;

Ptr<LPSIFT> LPSIFT::create(const std::vector<int>& windowSizes,
                           const float linearNoiseAlpha) {
    return makePtr<LPSIFT>(windowSizes, linearNoiseAlpha);
}

LPSIFT::LPSIFT(const std::vector<int>& windowSizes,
               const float linearNoiseAlpha)
    : windowSizes_(windowSizes),
      linearNoiseAlpha_(linearNoiseAlpha) {}

String LPSIFT::getDefaultName() const {
    return "Feature2D.LPSIFT";
}

// Step 1: Image Preprocessing
// Adds alpha * (y * cols + x) to each pixel to break flat plateaus deterministically.
void LPSIFT::addLinearRamp(Mat& image) const {
    if (linearNoiseAlpha_ <= 0.0f || image.empty()) return;

    Mat ramp(image.rows, image.cols, CV_32F);
    float* data = ramp.ptr<float>();
    std::iota(data, data + ramp.total(), 0.0f);
    ramp *= linearNoiseAlpha_;
    image += ramp;
}

// Check if keypoint value is unique in 3x3 neighborhood
// Based on MATLAB Test_feature_points_DE.m lines 117-154
bool LPSIFT::isUniqueInNeighborhood(const cv::Mat& image, int x, int y) const {
    const int rows = image.rows;
    const int cols = image.cols;

    // Get the value at the keypoint location
    float kptValue = image.at<float>(y, x);

    // Define neighborhood bounds with boundary checks
    int beginY = std::max(0, y - 1);
    int endY = std::min(rows - 1, y + 1);
    int beginX = std::max(0, x - 1);
    int endX = std::min(cols - 1, x + 1);

    // Count pixels with the same value in the neighborhood
    int count = 0;
    for (int ny = beginY; ny <= endY; ++ny) {
        for (int nx = beginX; nx <= endX; ++nx) {
            if (image.at<float>(ny, nx) == kptValue) {
                count++;
            }
        }
    }

    // Keypoint is unique if only one pixel (itself) has this value
    return count == 1;
}

bool LPSIFT::addKeypointCandidate(int x,
                                  int y,
                                  int windowSize,
                                  int octaveIndex,
                                  float response,
                                  int cols,
                                  int rows,
                                  std::vector<KeyPoint>& out) const {
    if (x < 0 || y < 0 || x >= cols || y >= rows || windowSize <= 0) return false;

    const float size = static_cast<float>(windowSize);
    KeyPoint kp(Point2f(static_cast<float>(x), static_cast<float>(y)), size);
    kp.response = response;
    kp.angle = -1.0f;
    kp.octave = octaveIndex;
    kp.class_id = windowSize;
    out.push_back(kp);
    return true;
}

// Feature Point Detection with 3x3 uniqueness filtering
void LPSIFT::detect(InputArray image,
                    std::vector<KeyPoint>& keypoints,
                    InputArray mask) {
    (void)mask;
    keypoints.clear();

    if (image.empty()) return;

    const Mat src = image.getMat();

    Mat gray;
    if (src.channels() > 1) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // Keep original for uniqueness check
    Mat originalGray;
    gray.convertTo(originalGray, CV_32F);

    // Add linear ramp for extrema detection
    Mat grayWithRamp;
    gray.convertTo(grayWithRamp, CV_32F);
    addLinearRamp(grayWithRamp);

    const int rows = gray.rows;
    const int cols = gray.cols;

    for (size_t idx = 0; idx < windowSizes_.size(); ++idx) {
        const int L = windowSizes_[idx];
        for (int y = 0; y < rows; y += L) {
            const int h = std::min(L, rows - y);
            for (int x = 0; x < cols; x += L) {
                const int w = std::min(L, cols - x);
                Rect roi(x, y, w, h);

                Mat tile = grayWithRamp(roi);
                double minVal = 0.0, maxVal = 0.0;
                Point minLoc, maxLoc;

                minMaxLoc(tile, &minVal, &maxVal, &minLoc, &maxLoc);

                const int gxMax = x + maxLoc.x;
                const int gyMax = y + maxLoc.y;
                const int gxMin = x + minLoc.x;
                const int gyMin = y + minLoc.y;
                const float response = static_cast<float>(maxVal - minVal);

                // Apply 3x3 uniqueness filter (from MATLAB code)
                if (isUniqueInNeighborhood(originalGray, gxMax, gyMax)) {
                    addKeypointCandidate(gxMax, gyMax, L, static_cast<int>(idx), response, cols, rows, keypoints);
                }
                if (isUniqueInNeighborhood(originalGray, gxMin, gyMin)) {
                    addKeypointCandidate(gxMin, gyMin, L, static_cast<int>(idx), response, cols, rows, keypoints);
                }
            }
        }
    }
}

// Compute custom 64-dimensional descriptor for a single keypoint
// Based on MATLAB Compute_descriptors.m
void LPSIFT::computeDescriptor(const cv::Mat& gray,
                               const cv::KeyPoint& kpt,
                               float* descriptor) const {
    const int d = SPATIAL_BINS;  // 4
    const int n = ORIENT_BINS;   // 4

    // Initialize descriptor to zero
    std::fill(descriptor, descriptor + DESCRIPTOR_SIZE, 0.0f);

    // Get keypoint info
    const float kptX = kpt.pt.x;
    const float kptY = kpt.pt.y;
    const int windowSize = kpt.class_id;  // The interrogation window size
    const int scale = static_cast<int>(std::log2(windowSize));
    const float s_scale = static_cast<float>(windowSize);

    // Calculate block indices
    const int block_j = static_cast<int>(kptX / s_scale);  // column block
    const int block_i = static_cast<int>(kptY / s_scale);  // row block

    // Calculate keypoint position within block
    const float kpt_w = kptX - block_j * s_scale;
    const float kpt_h = kptY - block_i * s_scale;

    // Extract the relevant block from the image
    const int blockStartX = static_cast<int>(block_j * s_scale);
    const int blockStartY = static_cast<int>(block_i * s_scale);
    const int blockEndX = std::min(blockStartX + windowSize, gray.cols);
    const int blockEndY = std::min(blockStartY + windowSize, gray.rows);
    const int blockW = blockEndX - blockStartX;
    const int blockH = blockEndY - blockStartY;

    if (blockW <= 2 || blockH <= 2) return;

    // Calculate histogram width and radius (from MATLAB)
    const float hist_width = 3.0f * std::sqrt(static_cast<float>(scale));
    int radius = static_cast<int>(std::round(hist_width * d));
    radius = std::min(radius, static_cast<int>(std::floor(std::sqrt(
        static_cast<float>(blockH * blockH + blockW * blockW)))));

    if (radius < d) radius = d;  // Ensure minimum radius

    const float radiusPerBin = static_cast<float>(radius) / d;

    // Loop through each spatial bin
    for (int ii = 0; ii < d; ++ii) {
        for (int jj = 0; jj < d; ++jj) {
            float hist[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            // Loop through pixels in this sub-region
            for (float i = -radiusPerBin; i <= radiusPerBin; i += 1.0f) {
                for (float j = -radiusPerBin; j <= radiusPerBin; j += 1.0f) {
                    // Calculate image coordinates
                    int img_h = static_cast<int>(std::floor(kpt_h + i + ii * radiusPerBin));
                    int img_w = static_cast<int>(std::floor(kpt_w + j + jj * radiusPerBin));

                    // Convert to global coordinates
                    int globalY = blockStartY + img_h;
                    int globalX = blockStartX + img_w;

                    // Check bounds (need 1 pixel margin for gradient computation)
                    if (globalY > 0 && globalY < gray.rows - 1 &&
                        globalX > 0 && globalX < gray.cols - 1) {
                        // Compute gradients (from MATLAB)
                        float dx = static_cast<float>(gray.at<uchar>(globalY, globalX + 1)) -
                                   static_cast<float>(gray.at<uchar>(globalY, globalX - 1));
                        float dy = static_cast<float>(gray.at<uchar>(globalY - 1, globalX)) -
                                   static_cast<float>(gray.at<uchar>(globalY + 1, globalX));

                        // Accumulate into 4 bins: +dx, +dy, -dx, -dy
                        if (dx >= 0) {
                            hist[0] += dx;
                        } else {
                            hist[2] += dx;  // Note: dx is negative here
                        }

                        if (dy >= 0) {
                            hist[1] += dy;
                        } else {
                            hist[3] += dy;  // Note: dy is negative here
                        }
                    }
                }
            }

            // Place histogram into descriptor
            int descIdx = (ii * d + jj) * n;
            for (int k = 0; k < n; ++k) {
                descriptor[descIdx + k] = hist[k];
            }
        }
    }

    // Normalize descriptor (L2 norm, threshold, re-normalize)
    float norm = 0.0f;
    for (int i = 0; i < DESCRIPTOR_SIZE; ++i) {
        norm += descriptor[i] * descriptor[i];
    }
    norm = std::sqrt(norm);

    if (norm > 1e-7f) {
        float threshold = 0.2f * norm;
        for (int i = 0; i < DESCRIPTOR_SIZE; ++i) {
            descriptor[i] = std::min(descriptor[i], threshold);
            descriptor[i] = std::max(descriptor[i], -threshold);
        }

        // Re-normalize
        norm = 0.0f;
        for (int i = 0; i < DESCRIPTOR_SIZE; ++i) {
            norm += descriptor[i] * descriptor[i];
        }
        norm = std::sqrt(norm);

        if (norm > 1e-7f) {
            for (int i = 0; i < DESCRIPTOR_SIZE; ++i) {
                descriptor[i] /= norm;
            }
        }
    }
}

void LPSIFT::compute(InputArray image,
                     std::vector<KeyPoint>& keypoints,
                     OutputArray descriptors) {
    if (keypoints.empty()) {
        descriptors.release();
        return;
    }

    Mat src = image.getMat();
    if (src.empty()) {
        descriptors.release();
        return;
    }

    Mat gray;
    if (src.channels() > 1) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src;
    }

    // Ensure 8-bit for gradient computation
    if (gray.type() != CV_8U) {
        gray.convertTo(gray, CV_8U);
    }

    // Create output descriptor matrix
    descriptors.create(static_cast<int>(keypoints.size()), DESCRIPTOR_SIZE, CV_32F);
    Mat desc = descriptors.getMat();

    // Compute descriptor for each keypoint
    for (size_t i = 0; i < keypoints.size(); ++i) {
        computeDescriptor(gray, keypoints[i], desc.ptr<float>(static_cast<int>(i)));
    }
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

// ============================================================================
// SIFTWithLPDescriptor Implementation
// Uses SIFT detector with LP-SIFT's custom 64-dim descriptor
// ============================================================================

Ptr<SIFTWithLPDescriptor> SIFTWithLPDescriptor::create() {
    return makePtr<SIFTWithLPDescriptor>();
}

SIFTWithLPDescriptor::SIFTWithLPDescriptor()
    : siftDetector_(SIFT::create()) {}

String SIFTWithLPDescriptor::getDefaultName() const {
    return "Feature2D.SIFTWithLPDescriptor";
}

void SIFTWithLPDescriptor::detect(InputArray image,
                                   std::vector<KeyPoint>& keypoints,
                                   InputArray mask) {
    siftDetector_->detect(image, keypoints, mask);
}

// Compute LP-SIFT style descriptor for SIFT keypoints
void SIFTWithLPDescriptor::computeDescriptor(const cv::Mat& gray,
                                              const cv::KeyPoint& kpt,
                                              float* descriptor) const {
    const int d = LPSIFT::SPATIAL_BINS;  // 4
    const int n = LPSIFT::ORIENT_BINS;   // 4

    std::fill(descriptor, descriptor + LPSIFT::DESCRIPTOR_SIZE, 0.0f);

    const float kptX = kpt.pt.x;
    const float kptY = kpt.pt.y;
    // Use keypoint size as the scale (SIFT provides this)
    const float kptSize = std::max(kpt.size, 4.0f);

    // Calculate radius based on keypoint size (similar to MATLAB hist_width calculation)
    int radius = static_cast<int>(std::round(kptSize * 2));
    if (radius < d) radius = d;

    const float radiusPerBin = static_cast<float>(radius) / d;

    // Loop through each spatial bin
    for (int ii = 0; ii < d; ++ii) {
        for (int jj = 0; jj < d; ++jj) {
            float hist[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (float i = -radiusPerBin; i <= radiusPerBin; i += 1.0f) {
                for (float j = -radiusPerBin; j <= radiusPerBin; j += 1.0f) {
                    int globalY = static_cast<int>(std::floor(kptY + i + (ii - d/2) * radiusPerBin));
                    int globalX = static_cast<int>(std::floor(kptX + j + (jj - d/2) * radiusPerBin));

                    if (globalY > 0 && globalY < gray.rows - 1 &&
                        globalX > 0 && globalX < gray.cols - 1) {
                        float dx = static_cast<float>(gray.at<uchar>(globalY, globalX + 1)) -
                                   static_cast<float>(gray.at<uchar>(globalY, globalX - 1));
                        float dy = static_cast<float>(gray.at<uchar>(globalY - 1, globalX)) -
                                   static_cast<float>(gray.at<uchar>(globalY + 1, globalX));

                        if (dx >= 0) {
                            hist[0] += dx;
                        } else {
                            hist[2] += dx;
                        }

                        if (dy >= 0) {
                            hist[1] += dy;
                        } else {
                            hist[3] += dy;
                        }
                    }
                }
            }

            int descIdx = (ii * d + jj) * n;
            for (int k = 0; k < n; ++k) {
                descriptor[descIdx + k] = hist[k];
            }
        }
    }

    // Normalize descriptor
    float norm = 0.0f;
    for (int i = 0; i < LPSIFT::DESCRIPTOR_SIZE; ++i) {
        norm += descriptor[i] * descriptor[i];
    }
    norm = std::sqrt(norm);

    if (norm > 1e-7f) {
        float threshold = 0.2f * norm;
        for (int i = 0; i < LPSIFT::DESCRIPTOR_SIZE; ++i) {
            descriptor[i] = std::min(descriptor[i], threshold);
            descriptor[i] = std::max(descriptor[i], -threshold);
        }

        norm = 0.0f;
        for (int i = 0; i < LPSIFT::DESCRIPTOR_SIZE; ++i) {
            norm += descriptor[i] * descriptor[i];
        }
        norm = std::sqrt(norm);

        if (norm > 1e-7f) {
            for (int i = 0; i < LPSIFT::DESCRIPTOR_SIZE; ++i) {
                descriptor[i] /= norm;
            }
        }
    }
}

void SIFTWithLPDescriptor::compute(InputArray image,
                                    std::vector<KeyPoint>& keypoints,
                                    OutputArray descriptors) {
    if (keypoints.empty()) {
        descriptors.release();
        return;
    }

    Mat src = image.getMat();
    if (src.empty()) {
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

    descriptors.create(static_cast<int>(keypoints.size()), LPSIFT::DESCRIPTOR_SIZE, CV_32F);
    Mat desc = descriptors.getMat();

    for (size_t i = 0; i < keypoints.size(); ++i) {
        computeDescriptor(gray, keypoints[i], desc.ptr<float>(static_cast<int>(i)));
    }
}

void SIFTWithLPDescriptor::detectAndCompute(InputArray image,
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
