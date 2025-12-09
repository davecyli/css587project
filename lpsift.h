/*
* David Li, Ben Schipunov, Kris Yu
 * CSS 587 - Final Project: LP-SIFT
 *
 * lpsift.h
 * LP-SIFT implementation based on:
 * Hao Li et al., "Local-peak scale-invariant feature transform for fast and random image stitching"
 * (arXiv:2405.08578v2).
 */

#ifndef LPSIFT_H
#define LPSIFT_H

#include <opencv2/features2d.hpp>
#include <vector>


class LPSIFT final : public cv::Feature2D {
public:
    // Different window sizes to cover good range of potential feature sizes in images
    static inline const std::vector<int> DEFAULT_WINDOW_SIZES = { 16, 32, 64, 128, 256 };
    static constexpr float DEFAULT_LINEAR_NOISE_ALPHA = 1e-6f; // Sufficiently small noise constant

    /** @brief Factory for an LPSIFT detector/descriptor.
     *  @param windowSizes Interrogation window sizes (non-empty, values > 1).
     *  @param linearNoiseAlpha Small ramp magnitude added during preprocessing.
     *  @return Pointer created via cv::makePtr.
     */
    static cv::Ptr<LPSIFT> create(
        const std::vector<int>& windowSizes = DEFAULT_WINDOW_SIZES,
        float linearNoiseAlpha = DEFAULT_LINEAR_NOISE_ALPHA);

    /** @brief Construct an LPSIFT detector/descriptor.
     *  Public to allow cv::makePtr; defaults are defined only on create().
     *  @param windowSizes Interrogation window sizes (non-empty, values > 1).
     *  @param linearNoiseAlpha Ramp magnitude applied during preprocessing.
     */
    explicit LPSIFT(const std::vector<int>& windowSizes,
                    float linearNoiseAlpha);

    /** @brief OpenCV registry name for this implementation. */
    cv::String getDefaultName() const override; // NOLINT(modernize-use-nodiscard) matching OpenCV base signature
    /** @brief Dimension of the descriptor (delegates to SIFT). */
    [[nodiscard]] int descriptorSize() const override { return descriptor_->descriptorSize(); }
    /** @brief OpenCV type of the descriptor matrix (delegates to SIFT). */
    [[nodiscard]] int descriptorType() const override { return descriptor_->descriptorType(); }

    /** @brief Detect keypoints via Local Peaks after linear ramp preprocessing.
     *  @param image Input image.
     *  @param keypoints Output vector of detected keypoints.
     *  @param mask Mask input (ignored; kept for API compatibility).
     */
    void detect(cv::InputArray image,
                std::vector<cv::KeyPoint>& keypoints,
                cv::InputArray mask) override;

    /** @brief Compute SIFT descriptors for provided keypoints.
     *  @param image Input image.
     *  @param keypoints Keypoints to describe (must be non-empty).
     *  @param descriptors Output descriptor matrix.
     */
    void compute(cv::InputArray image,
                 std::vector<cv::KeyPoint>& keypoints,
                 cv::OutputArray descriptors) override;

    /** @brief Combined detect and compute pipeline. Calls LPSIFT detect then SIFT compute.
     *  @param image Input image.
     *  @param mask Optional mask (ignored).
     *  @param keypoints Output keypoints (or input when useProvidedKeypoints is true).
     *  @param descriptors Output descriptor matrix.
     *  @param useProvidedKeypoints If false, runs detect() first; otherwise only computes descriptors.
     */
    void detectAndCompute(cv::InputArray image,
                          cv::InputArray mask,
                          std::vector<cv::KeyPoint>& keypoints,
                          cv::OutputArray descriptors,
                          bool useProvidedKeypoints) override;

private:
    cv::Ptr<cv::Feature2D> descriptor_; // Pointer to SIFT instance for descriptor implementation
    std::vector<int> windowSizes_;
    float linearNoiseAlpha_;


    /** @brief Adds a linear ramp to the image
     *
     * Note: Uses formular alpha * (y * cols + x) ramp to make pixel values strictly increasing.
     * @param image Input Image
     */
    void addLinearRamp(cv::Mat& image) const;

    /** @brief Validate bounds and append a keypoint candidate.
     *  @param x Pixel x coordinate.
     *  @param y Pixel y coordinate.
     *  @param windowSize Interrogation window size (pixels).
     *  @param octaveIndex Index of the current scale/octave.
     *  @param response Response/contrast value to store.
     *  @param cols Image width (used for bounds check).
     *  @param rows Image height (used for bounds check).
     *  @param out Destination vector to receive the keypoint.
     *  @return True if the keypoint was within bounds and added; false otherwise.
     */
    static bool addKeypointCandidate(int x,
                                     int y,
                                     int windowSize,
                                     int octaveIndex,
                                     float response,
                                     int cols,
                                     int rows,
                                     std::vector<cv::KeyPoint>& out);
};

#endif //LPSIFT_H
