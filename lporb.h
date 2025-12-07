#ifndef LPORB_H
#define LPORB_H

#include <opencv2/features2d.hpp>
#include <vector>

using namespace cv;

class LPORB final : public Feature2D {
public:
    static Ptr<LPORB> create(
        const std::vector<int>& windowSizes = { 16, 32, 64, 128, 256 },
        float linearNoiseAlpha = 1e-6f);

    // Public to allow cv::makePtr; defaults are defined only on create().
    explicit LPORB(const std::vector<int>& windowSizes,
                    float linearNoiseAlpha);

    String getDefaultName() const override; // NOLINT(modernize-use-nodiscard) matching OpenCV base signature

    [[nodiscard]] int descriptorSize() const override { return descriptor_->descriptorSize(); }
    [[nodiscard]] int descriptorType() const override { return descriptor_->descriptorType(); }

    void detect(InputArray image,
                std::vector<KeyPoint>& keypoints,
                InputArray mask) override;

    void compute(InputArray image,
                 std::vector<KeyPoint>& keypoints,
                 OutputArray descriptors) override;

    void detectAndCompute(InputArray image,
                          InputArray mask,
                         std::vector<KeyPoint>& keypoints,
                         OutputArray descriptors,
                         bool useProvidedKeypoints) override;

private:
    Ptr<Feature2D> descriptor_; // Descriptor implementation (ORB-backed)
    std::vector<int> windowSizes_;
    float linearNoiseAlpha_;

    // Adds alpha * (y * cols + x) ramp to make pixel values strictly increasing; expects CV_32F input.
    void addLinearRamp(Mat& image) const;
    static bool addKeypointCandidate(int x,
                              int y,
                              int windowSize,
                              int octaveIndex,
                              float response,
                              int cols,
                              int rows,
                              std::vector<KeyPoint>& out) ;
};

#endif //LPORB_H
