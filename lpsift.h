#ifndef LPSIFT_H
#define LPSIFT_H

#include <opencv2/features2d.hpp>
#include <vector>

class LPSIFT final : public cv::Feature2D {
public:
    static cv::Ptr<LPSIFT> create(
        const std::vector<int>& windowSizes = {8, 32},
        float linearNoiseAlpha = 1e-6f,
        float beta0 = 0.1f,
        int subregionGrid = 4);

    // Public to allow cv::makePtr; defaults are defined only on create().
    explicit LPSIFT(const std::vector<int>& windowSizes,
                    float linearNoiseAlpha,
                    float beta0,
                    int subregionGrid);

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
    float beta0_;
    int subregionGrid_;

    float computePatchSize(int windowSize) const;
    void addLinearRamp(cv::Mat& image) const;
};

#endif //LPSIFT_H
