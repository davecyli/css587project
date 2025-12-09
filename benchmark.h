/*
 * David Li, Ben Schipunov, Kris Yu
 * CSS 587 - Final Project: LP-SIFT
 *
 * benchmark.h
 * Benchmarking framework for comparing feature detection algorithms.
 * Metrics based on LP-SIFT paper (arXiv:2405.08578v2) Table 2.
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <string>
#include <vector>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <set>
#include <map>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

namespace fs = std::filesystem;

using namespace cv;
using namespace std;

// ============================================================================
// Constants
// ============================================================================

// Maximum keypoints for BFMatcher (IMGIDX_ONE limit ~65536)
// FLANN matcher has no practical limit
constexpr int MAX_KEYPOINTS_BF = 50000;

// Matcher types
enum class MatcherType {
    BRUTE_FORCE,  // Exact matching, limited to ~65k keypoints
    FLANN         // Approximate matching, handles millions of keypoints
};

// Minimum matches required for homography estimation
constexpr size_t MIN_MATCHES = 4;

// RANSAC parameters
constexpr double RANSAC_THRESHOLD = 3.0;
constexpr int RNG_SEED = 12345;

// ============================================================================
// Image Size Category
// ============================================================================

// Image size category based on paper's classification
enum class ImageSizeCategory {
    SMALL,   // < 1 MP (e.g., 602x400, 653x490)
    MEDIUM,  // 1-3 MP (e.g., 1024x768, 1080x1920)
    LARGE    // > 3 MP (e.g., 3072x4096)
};

// Converts ImageSizeCategory to string
inline std::string imageSizeCategoryToString(ImageSizeCategory cat) {
    switch (cat) {
        case ImageSizeCategory::SMALL: return "Small";
        case ImageSizeCategory::MEDIUM: return "Medium";
        case ImageSizeCategory::LARGE: return "Large";
        default: return "Unknown";
    }
}

// Determines image size category based on pixel count
inline ImageSizeCategory getImageSizeCategory(int width, int height) {
    long pixels = static_cast<long>(width) * height;
    if (pixels < 1000000) return ImageSizeCategory::SMALL;       // < 1 MP
    if (pixels < 3000000) return ImageSizeCategory::MEDIUM;      // 1-3 MP
    return ImageSizeCategory::LARGE;                              // > 3 MP
}

inline std::vector<int> getWindowSize(int width, int height) {
    ImageSizeCategory category = getImageSizeCategory(width, height);
    return { 16, 32, 64, 128, 256 };
    switch (category) {
        case ImageSizeCategory::SMALL:
            return {32, 40};
        case ImageSizeCategory::MEDIUM:
            return {32, 64};
        case ImageSizeCategory::LARGE:
            return {256, 512};
        default:
            return {32, 64};
    }
}

// ============================================================================
// StitchingMetrics - Performance metrics structure
// ============================================================================

/*
 * StitchingMetrics - Comprehensive metrics structure for performance evaluation
 * Based on Table 2 from the LP-SIFT paper
 */
struct StitchingMetrics {
    // Dataset info
    std::string datasetName;
    std::string algorithmName;
    ImageSizeCategory sizeCategory = ImageSizeCategory::SMALL;

    // Image resolutions (Width x Height in pixels)
    int referenceWidth = 0;
    int referenceHeight = 0;
    int registeredWidth = 0;
    int registeredHeight = 0;

    // Feature detection metrics
    int numKeypointsReference = 0;
    int numKeypointsRegistered = 0;
    int numMatches = 0;
    int numInliers = 0;

    // Timing metrics (in seconds with 1/100 precision as per paper)
    double detectionTimeReference = 0.0;
    double detectionTimeRegistered = 0.0;
    double descriptorTimeReference = 0.0;
    double descriptorTimeRegistered = 0.0;
    double matchingTime = 0.0;
    double homographyTime = 0.0;
    double warpingTime = 0.0;
    double totalStitchingTime = 0.0;

    cv::Mat homography;  // Estimated homography matrix
    cv::Mat baselineH;

    // LP-SIFT specific parameters
    std::string windowSizes;

    // Stitching success
    bool stitchingSuccess = false;
    std::string failureReason;

    // Quality metrics (optional)
    double reprojectionError = 0.0;

    // Get resolution string
    std::string getReferenceResolution() const {
        return std::to_string(referenceWidth) + "x" + std::to_string(referenceHeight);
    }

    std::string getRegisteredResolution() const {
        return std::to_string(registeredWidth) + "x" + std::to_string(registeredHeight);
    }

    // Format time to 2 decimal places (1/100 precision as per paper)
    static std::string formatTime(double seconds) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << seconds;
        return oss.str();
    }

    static std::string printHomography(const cv::Mat& H) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4);

        oss << "[";

        for (int i = 0; i < H.rows; i++) {
            oss << "[";

            for (int j = 0; j < H.cols; j++) {
                oss << H.at<double>(i, j);
                if (j < H.cols - 1)
                    oss << ", ";
            }

            oss << "]";
            if (i < H.rows - 1)
                oss << ", ";
        }

        oss << "]";

        return oss.str();
    }
};

// ============================================================================
// Timer - High-resolution timer for benchmarking
// ============================================================================

class Timer {
public:
    void start() {
        startTime_ = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        endTime_ = std::chrono::high_resolution_clock::now();
    }

    double elapsedSeconds() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime_ - startTime_);
        return duration.count() / 1000000.0;
    }

    double elapsedMilliseconds() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime_ - startTime_);
        return duration.count() / 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point startTime_;
    std::chrono::high_resolution_clock::time_point endTime_;
};

// ============================================================================
// CSVExporter - Exports benchmark results to CSV format
// ============================================================================

class CSVExporter {
public:
    explicit CSVExporter(const std::string& filename) : filename_(filename) {}

    void writeHeader();
    void writeMetrics(const StitchingMetrics& m);
    void writeAllMetrics(const std::vector<StitchingMetrics>& metrics);

private:
    std::string filename_;
};

// ============================================================================
// BenchmarkRunner - Automated testing framework for comparing algorithms
// ============================================================================

class BenchmarkRunner {
public:
    // Detector configuration
    struct DetectorConfig {
        std::string name;
        cv::Ptr<cv::Feature2D> detector;
        cv::NormTypes matcherNorm;
        MatcherType matcherType = MatcherType::FLANN;  // Default to FLANN for no keypoint limit
    };

    struct DetectorFilter {
        bool SIFT;
        bool ORB;
        bool BRISK;
        bool SURF;
        bool LPSIFT;
        bool LPORB;
    };

    cv::Mat baselineH;

    BenchmarkRunner() = default;

    // Add a detector to benchmark
    void addDetector(const std::string& name,
                     cv::Ptr<cv::Feature2D> detector,
                     cv::NormTypes norm,
                     MatcherType matcherType = MatcherType::FLANN);

    void clearDetectors();

    // Run benchmark on a single image pair
    StitchingMetrics runSingleBenchmark(
        const std::string& datasetName,
        const cv::Mat& referenceImg,
        const cv::Mat& registeredImg,
        const DetectorConfig& config,
		const vector<int>& lpsiftWindowSizes,
        const std::string& outputPath);

    // Run benchmark on all detectors for a single image pair
    std::vector<StitchingMetrics> runAllDetectors(
        const std::string& datasetName,
        const cv::Mat& referenceImg,
        const cv::Mat& registeredImg,
		const vector<int>& lpsiftWindowSizes,
        const std::string& outputPath
        );

    // Run benchmark on all image sets in a directory
    std::vector<StitchingMetrics> runOnDirectory(
        const string& imageDir,
        const set<string>& filteredImageSets,
		const map<string, DetectorFilter>& filteredDetectors,
        const string& outputPath);

    // Print summary table (similar to paper's Table 2)
    static void printSummaryTable(const std::vector<StitchingMetrics>& results);

private:
    std::vector<DetectorConfig> detectors_;

    // Warp and blend images using homography
    static cv::Mat warpAndBlend(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& H);
};

#endif // BENCHMARK_H
