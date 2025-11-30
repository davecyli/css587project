#ifndef LPSIFT_H
#define LPSIFT_H

#include <opencv2/features2d.hpp>

class LPSIFT final : public cv::Feature2D {
public:
    static cv::Ptr<LPSIFT> create();

    // Public to allow cv::makePtr;
    LPSIFT();

    cv::String getDefaultName() const override; // NOLINT(modernize-use-nodiscard) matching OpenCV base signature

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
    cv::Ptr<cv::Feature2D> descriptor_; // placeholder for SIFT descriptor
};

#endif //LPSIFT_H
