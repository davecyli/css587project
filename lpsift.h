#ifndef LPSIFT_H
#define LPSIFT_H

#include <opencv2/features2d.hpp>
#include <vector>

class LPSIFT final : public cv::Feature2D {
public:
    static cv::Ptr<LPSIFT> create(
        const std::vector<int>& windowSizes = {16, 40, 128},
        float linearNoiseAlpha = 1e-6f);

    // Public to allow cv::makePtr; defaults are defined only on create().
    explicit LPSIFT(const std::vector<int>& windowSizes,
                    float linearNoiseAlpha);

    cv::String getDefaultName() const override; // NOLINT(modernize-use-nodiscard) matching OpenCV base signature

    int descriptorSize() const override { return descriptor_->descriptorSize(); }
    int descriptorType() const override { return descriptor_->descriptorType(); }

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
    cv::Ptr<cv::Feature2D> descriptor_; // SIFT descriptor implementation
    std::vector<int> windowSizes_;
    float linearNoiseAlpha_;

    // Adds alpha * (y * cols + x) ramp to make pixel values strictly increasing; expects CV_32F input.
    void addLinearRamp(cv::Mat& image) const;
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
