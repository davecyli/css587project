/*
 * LP-SIFT skeleton: minimal Feature2D subclass with stubbed detection and
 * OpenCV SIFT descriptors. Detection currently returns no keypoints; fill in
 * with the paper's algorithm as needed.
 */

#include "lpsift.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;

LPSIFT::LPSIFT()
    : descriptor_(SIFT::create()) {}

Ptr<LPSIFT> LPSIFT::create() {
    return makePtr<LPSIFT>();
}

String LPSIFT::getDefaultName() const {
    return "Feature2D.LPSIFT";
}

void LPSIFT::detect(InputArray image,
                    std::vector<KeyPoint>& keypoints,
                    InputArray mask) {
    CV_UNUSED(image);
    CV_UNUSED(mask);
    keypoints.clear();
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

    descriptor_->compute(gray, keypoints, descriptors);
}

void LPSIFT::detectAndCompute(InputArray image,
                              InputArray mask,
                              std::vector<KeyPoint>& keypoints,
                              OutputArray descriptors,
                              bool useProvidedKeypoints) {
    if (!useProvidedKeypoints) {
        detect(image, keypoints, mask);
    }

    if (keypoints.empty()) {
        descriptors.release();
        return;
    }

    compute(image, keypoints, descriptors);
}
