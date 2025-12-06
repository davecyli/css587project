#ifndef LPSIFT_H
#define LPSIFT_H

#include <opencv2/features2d.hpp>
#include <vector>

/*
 * LP-SIFT: Local-Peak Scale-Invariant Feature Transform
 *
 * Based on: Hao Li et al., "Local-peak scale-invariant feature transform
 * for fast and random image stitching" (arXiv:2405.08578v2)
 *
 * Key differences from standard SIFT:
 * - No Gaussian pyramid construction (major speedup)
 * - Custom 64-dimensional descriptor (4x4 spatial bins x 4 gradient bins)
 * - Simple dx/dy gradient computation
 */
class LPSIFT final : public cv::Feature2D {
public:
    // Descriptor parameters matching the paper
    static constexpr int SPATIAL_BINS = 4;      // d = 4 spatial bins per dimension
    static constexpr int ORIENT_BINS = 4;       // n = 4 orientation bins (+dx, +dy, -dx, -dy)
    static constexpr int DESCRIPTOR_SIZE = SPATIAL_BINS * SPATIAL_BINS * ORIENT_BINS; // 64

    static cv::Ptr<LPSIFT> create(
        const std::vector<int>& windowSizes = {16, 40, 128},
        float linearNoiseAlpha = 1e-6f);

    explicit LPSIFT(const std::vector<int>& windowSizes,
                    float linearNoiseAlpha);

    cv::String getDefaultName() const override;

    int descriptorSize() const override { return DESCRIPTOR_SIZE; }  // 64
    int descriptorType() const override { return CV_32F; }

    void detect(cv::InputArray image,
                std::vector<cv::KeyPoint>& keypoints,
                cv::InputArray mask) override;

    void compute(cv::InputArray image,
                 std::vector<cv::KeyPoint>& keypoints,
                 cv::OutputArray descriptors) override;

    void detectAndCompute(cv::InputArray image,
                          cv::InputArray mask,
                          std::vector<cv::KeyPoint>& keypoints,
                          cv::OutputArray descriptors,
                          bool useProvidedKeypoints) override;

private:
    std::vector<int> windowSizes_;
    float linearNoiseAlpha_;

    // Adds alpha * (y * cols + x) ramp to break flat plateaus
    void addLinearRamp(cv::Mat& image) const;

    // Check if keypoint is unique in 3x3 neighborhood (from original MATLAB)
    bool isUniqueInNeighborhood(const cv::Mat& image, int x, int y) const;

    // Compute single keypoint descriptor (64-dim)
    void computeDescriptor(const cv::Mat& gray,
                          const cv::KeyPoint& kpt,
                          float* descriptor) const;

    bool addKeypointCandidate(int x,
                              int y,
                              int windowSize,
                              int octaveIndex,
                              float response,
                              int cols,
                              int rows,
                              std::vector<cv::KeyPoint>& out) const;
};

#endif //LPSIFT_H
