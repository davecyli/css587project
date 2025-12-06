/*
 * David Li, Ben Schipunov, Kris Yu
 * CSS 587 - Final Project: LP-SIFT
 *
 * benchmark.cpp
 * Implementation of benchmarking framework classes.
 */

#include "benchmark.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "lpsift.h"

using namespace cv;
using namespace std;

// Helper function to limit keypoints by keeping the strongest ones
static void limitKeypoints(std::vector<cv::KeyPoint>& kpts, size_t maxCount) {
    if (kpts.size() > maxCount) {
        // Sort by response (strength) descending and keep top N
        std::sort(kpts.begin(), kpts.end(),
            [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                return a.response > b.response;
            });
        kpts.resize(maxCount);
    }
}

// ============================================================================
// CSVExporter Implementation
// ============================================================================

void CSVExporter::writeHeader() {
    std::ofstream file(filename_, std::ios::out | std::ios::trunc);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename_ << " for writing." << std::endl;
        return;
    }

    file << "Dataset,"
         << "Size Category,"
         << "Algorithm,"
         << "Reference Resolution,"
         << "Registered Resolution,"
         << "Keypoints (Reference),"
         << "Keypoints (Registered),"
         << "Matches,"
         << "Inliers,"
         << "Window Size (L),"
         << "Detection Time Ref (s),"
         << "Detection Time Reg (s),"
         << "Descriptor Time Ref (s),"
         << "Descriptor Time Reg (s),"
         << "Matching Time (s),"
         << "Homography Time (s),"
         << "Warping Time (s),"
         << "Total Stitching Time (s),"
         << "Success,"
         << "Homography Matrix,"
         << "Homography Difference from SIFT,"
         << "Failure Reason"
         << "\n";

    file.close();
}

namespace {

std::string escapeCsv(const std::string& s) {
    std::string r;
    for (char c : s)
        r += (c == '"') ? "\"\"" : std::string(1, c);
    return r;
}

template <typename T>
std::string toString(const T& value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

template <typename... Args>
std::string makeCsvRow(const Args... args) {
    std::vector<std::string> items = { escapeCsv(toString(args))... };

    std::ostringstream row;
    for (size_t i = 0; i < items.size(); ++i) {
        row << "\"" << items[i] << "\"";
        if (i + 1 < items.size())
            row << ",";
    }
    return row.str();
}

} // anonymous namespace

void CSVExporter::writeMetrics(const StitchingMetrics& m) {
    std::ofstream file(filename_, std::ios::out | std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename_ << " for appending." << std::endl;
        return;
    }

    std::string csvRow = makeCsvRow(
        m.datasetName,
        imageSizeCategoryToString(m.sizeCategory),
        m.algorithmName,
        m.getReferenceResolution(),
        m.getRegisteredResolution(),
        m.numKeypointsReference,
        m.numKeypointsRegistered,
        m.numMatches,
        m.numInliers,
        m.windowSizes,
        StitchingMetrics::formatTime(m.detectionTimeReference),
        StitchingMetrics::formatTime(m.detectionTimeRegistered),
        StitchingMetrics::formatTime(m.descriptorTimeReference),
        StitchingMetrics::formatTime(m.descriptorTimeRegistered),
        StitchingMetrics::formatTime(m.matchingTime),
        StitchingMetrics::formatTime(m.homographyTime),
        StitchingMetrics::formatTime(m.warpingTime),
        StitchingMetrics::formatTime(m.totalStitchingTime),
        m.printHomography(m.homography),
        m.printHomography(m.homography - m.baselineH),
        (m.stitchingSuccess ? "Yes" : "No"),
        m.failureReason
    );

    file << csvRow << "\n";

    file.close();
}

void CSVExporter::writeAllMetrics(const std::vector<StitchingMetrics>& metrics) {
    writeHeader();
    for (const auto& m : metrics) {
        writeMetrics(m);
    }
}

// ============================================================================
// BenchmarkRunner Implementation
// ============================================================================

void BenchmarkRunner::addDetector(const std::string& name,
                                   cv::Ptr<cv::Feature2D> detector,
                                   cv::NormTypes norm,
                                   MatcherType matcherType) {
    detectors_.push_back({name, detector, norm, matcherType});
}

void BenchmarkRunner::clearDetectors() {
    detectors_.clear();
}

void naiveBruteForceMatch(
    const cv::Mat& desc1,
    const cv::Mat& desc2,
    float threshold,
    vector<DMatch>& matches
)
{
    matches.clear();
    matches.reserve(50000);   // reserve memory (optional)

    for (int i = 0; i < desc1.rows; i++)
    {
        const float* d1 = desc1.ptr<float>(i);

        for (int j = 0; j < desc2.rows; j++)
        {
            float dist = cv::norm(desc1.row(i), desc2.row(j), cv::NORM_L2);

            if (dist < threshold)
            {
                matches.emplace_back(i, j, dist);
            }
        }
    }
}


string joinInts(const vector<int>& v) {
    std::ostringstream oss;
    for (size_t i = 0; i < v.size(); i++) {
        if (i > 0) oss << ",";
        oss << v[i];
    }
    return oss.str();
}

StitchingMetrics BenchmarkRunner::runSingleBenchmark(
    const std::string& datasetName,
    const cv::Mat& referenceImg,
    const cv::Mat& registeredImg,
    const DetectorConfig& config,
	const vector<int>& lpsiftWindowSizes,
    const std::string& outputPath
) {
    StitchingMetrics metrics;
    metrics.datasetName = datasetName;
    metrics.algorithmName = config.name;

    // Only set window sizes for LP-SIFT algorithm, use "x" for others
    if (config.name == "LP-SIFT") {
        metrics.windowSizes = joinInts(lpsiftWindowSizes);
    } else {
        metrics.windowSizes = "x";
    }

    // Record image dimensions
    metrics.referenceWidth = referenceImg.cols;
    metrics.referenceHeight = referenceImg.rows;
    metrics.registeredWidth = registeredImg.cols;
    metrics.registeredHeight = registeredImg.rows;
    metrics.sizeCategory = getImageSizeCategory(referenceImg.cols, referenceImg.rows);

    Timer totalTimer, stepTimer;
    totalTimer.start();

    try {
        // Convert to grayscale
        cv::Mat gray1, gray2;
        cv::cvtColor(referenceImg, gray1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(registeredImg, gray2, cv::COLOR_BGR2GRAY);

        // Feature detection - Reference image
        std::vector<cv::KeyPoint> kpts1, kpts2;
        cv::Mat desc1, desc2;

        stepTimer.start();
        config.detector->detect(gray1, kpts1);
        stepTimer.stop();
        metrics.detectionTimeReference = stepTimer.elapsedSeconds();
        metrics.numKeypointsReference = static_cast<int>(kpts1.size());

        // Feature detection - Registered image
        stepTimer.start();
        config.detector->detect(gray2, kpts2);
        stepTimer.stop();
        metrics.detectionTimeRegistered = stepTimer.elapsedSeconds();
        metrics.numKeypointsRegistered = static_cast<int>(kpts2.size());

        // Check for empty keypoints
        if (kpts1.empty() || kpts2.empty()) {
            metrics.stitchingSuccess = false;
            metrics.failureReason = "Empty keypoints";
            totalTimer.stop();
            metrics.totalStitchingTime = totalTimer.elapsedSeconds();
            return metrics;
        }

        // Limit keypoints only for BFMatcher (has ~65536 limit due to IMGIDX_ONE)
        // FLANN can handle unlimited keypoints
        /*if (config.matcherType == MatcherType::BRUTE_FORCE) {
            limitKeypoints(kpts1, MAX_KEYPOINTS_BF);
            limitKeypoints(kpts2, MAX_KEYPOINTS_BF);
            metrics.numKeypointsReference = static_cast<int>(kpts1.size());
            metrics.numKeypointsRegistered = static_cast<int>(kpts2.size());
        }*/

        // Descriptor computation - Reference image
        stepTimer.start();
        config.detector->compute(gray1, kpts1, desc1);
        stepTimer.stop();
        metrics.descriptorTimeReference = stepTimer.elapsedSeconds();

        // Descriptor computation - Registered image
        stepTimer.start();
        config.detector->compute(gray2, kpts2, desc2);
        stepTimer.stop();
        metrics.descriptorTimeRegistered = stepTimer.elapsedSeconds();

        // Update keypoint counts after potential filtering during compute
        metrics.numKeypointsReference = static_cast<int>(kpts1.size());
        metrics.numKeypointsRegistered = static_cast<int>(kpts2.size());

        // Check for empty descriptors
        if (desc1.empty() || desc2.empty()) {
            metrics.stitchingSuccess = false;
            metrics.failureReason = "Empty descriptors";
            totalTimer.stop();
            metrics.totalStitchingTime = totalTimer.elapsedSeconds();
            return metrics;
        }

        // Feature matching
        stepTimer.start();
        std::vector<cv::DMatch> matches;

        //if (config.matcherType == MatcherType::FLANN) {
        //    // FLANN matcher - can handle unlimited keypoints
        //    cv::Ptr<cv::DescriptorMatcher> matcher;

        //    if (config.matcherNorm == cv::NORM_HAMMING || config.matcherNorm == cv::NORM_HAMMING2) {
        //        // Binary descriptors (ORB, BRISK) - use LSH index
        //        cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2);
        //        cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(50);
        //        matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);
        //    } else {
        //        // Float descriptors (SIFT) - use KDTree index
        //        cv::Ptr<cv::flann::IndexParams> indexParams = cv::makePtr<cv::flann::KDTreeIndexParams>(5);
        //        cv::Ptr<cv::flann::SearchParams> searchParams = cv::makePtr<cv::flann::SearchParams>(50);
        //        matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams, searchParams);
        //    }

        //    // Use knnMatch with ratio test for better quality matches
        //    std::vector<std::vector<cv::DMatch>> knnMatches;
        //    matcher->knnMatch(desc1, desc2, knnMatches, 2);

        //    // Apply Lowe's ratio test
        //    const float ratioThresh = 0.75f;
        //    for (const auto& knn : knnMatches) {
        //        if (knn.size() >= 2 && knn[0].distance < ratioThresh * knn[1].distance) {
        //            matches.push_back(knn[0]);
        //        }
        //    }
        //} else {
        //    // BFMatcher - exact matching but limited to ~65k keypoints
        //    
        //}

        cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(config.matcherNorm);

        try {
            matcher->match(desc1, desc2, matches);
		}
		catch (exception& e) {
            metrics.stitchingSuccess = false;
            metrics.failureReason = std::string("Over size");
            totalTimer.stop();
            metrics.totalStitchingTime = totalTimer.elapsedSeconds();
            return metrics;
        }
        stepTimer.stop();
        metrics.matchingTime = stepTimer.elapsedSeconds();
        metrics.numMatches = static_cast<int>(matches.size());

        // Check for sufficient matches
        if (matches.size() < MIN_MATCHES) {
            metrics.stitchingSuccess = false;
            metrics.failureReason = "Insufficient matches (<4)";
            totalTimer.stop();
            metrics.totalStitchingTime = totalTimer.elapsedSeconds();
            return metrics;
        }

        // Extract matched points
        std::vector<cv::Point2f> pts1, pts2;
        for (const auto& m : matches) {
            pts1.push_back(kpts1[m.queryIdx].pt);
            pts2.push_back(kpts2[m.trainIdx].pt);
        }

        // RANSAC homography estimation
        stepTimer.start();
        cv::setRNGSeed(RNG_SEED);
        std::vector<uchar> inlierMask;
        cv::Mat H = cv::findHomography(pts2, pts1, cv::RANSAC, RANSAC_THRESHOLD, inlierMask);
        stepTimer.stop();
        metrics.homographyTime = stepTimer.elapsedSeconds();

        // Count inliers
        metrics.numInliers = cv::countNonZero(inlierMask);

        // Check for valid homography
        if (H.empty()) {
            metrics.stitchingSuccess = false;
            metrics.failureReason = "Homography computation failed";
            totalTimer.stop();
            metrics.totalStitchingTime = totalTimer.elapsedSeconds();
            return metrics;
        }

        // Image warping and blending
        stepTimer.start();
        cv::Mat stitched = warpAndBlend(registeredImg, referenceImg, H);
        stepTimer.stop();
        metrics.warpingTime = stepTimer.elapsedSeconds();

        totalTimer.stop();
        metrics.totalStitchingTime = totalTimer.elapsedSeconds();
        metrics.stitchingSuccess = true;

        metrics.homography = cv::Mat(H);

        if (config.name == "SIFT") {
            this->baselineH = cv::Mat(H);
        }

        metrics.baselineH = this->baselineH;

        // Save stitched image if requested
        if (!outputPath.empty()) {
            std::string outFile = outputPath + "/" + datasetName + "_" + config.name + "_stitched.jpg";
            cv::imwrite(outFile, stitched);
        }

    } catch (const std::exception& e) {
        metrics.stitchingSuccess = false;
        metrics.failureReason = std::string("Exception: ") + e.what();
        totalTimer.stop();
        metrics.totalStitchingTime = totalTimer.elapsedSeconds();
    }

    return metrics;
}

std::vector<StitchingMetrics> BenchmarkRunner::runAllDetectors(
    const std::string& datasetName,
    const cv::Mat& referenceImg,
    const cv::Mat& registeredImg,
	const vector<int>& windowSizes,
    const std::string& outputPath
) {
    std::vector<StitchingMetrics> results;

    for (const auto& config : detectors_) {
        std::cout << "  Running " << config.name << "..." << std::flush;

        StitchingMetrics metrics = runSingleBenchmark(
            datasetName, referenceImg, registeredImg, config, windowSizes,
            outputPath
        );

        if (metrics.stitchingSuccess) {
            std::cout << " Done (" << StitchingMetrics::formatTime(metrics.totalStitchingTime)
                      << "s, " << metrics.numKeypointsReference << "/"
                      << metrics.numKeypointsRegistered << " keypoints)" << std::endl;
        } else {
            std::cout << " Failed: " << metrics.failureReason << std::endl;
        }

        results.push_back(metrics);
    }

    return results;
}

std::vector<StitchingMetrics> BenchmarkRunner::runOnDirectory(
    const string& imageDir,
	const set<string>& filteredImageSets,
	const map<string, DetectorFilter>& filteredDetectors,
    const string& outputPath
) {
    std::vector<StitchingMetrics> allResults;

    if (!fs::exists(imageDir) || !fs::is_directory(imageDir)) {
        std::cerr << "Error: Image directory does not exist: " << imageDir << std::endl;
        return allResults;
    }

    // Collect and sort image set directories
    std::vector<std::string> imageSets;
    for (const auto& entry : fs::directory_iterator(imageDir)) {
        if (entry.is_directory()) {
            imageSets.push_back(entry.path().string());
        }
    }
    std::sort(imageSets.begin(), imageSets.end());

    for (const auto& setPath : imageSets) {
        std::string setName = fs::path(setPath).filename().string();

        if (filteredImageSets.empty() || filteredImageSets.find(setName) != filteredImageSets.end()) {
            
            std::cout << "\nProcessing: " << setName << std::endl;

            // Load images (registered.jpg and reference.jpg as per main.cpp convention)
            cv::Mat registered = cv::imread(setPath + "/registered.jpg");
            cv::Mat reference = cv::imread(setPath + "/reference.jpg");

            if (registered.empty() || reference.empty()) {
                std::cerr << "  Warning: Could not load images from " << setPath << std::endl;
                continue;
            }

            std::cout << "  Reference: " << reference.cols << "x" << reference.rows
                << ", Registered: " << registered.cols << "x" << registered.rows << std::endl;

            clearDetectors(); // Clear previous detectors if any

            bool globalFilters = filteredDetectors.count("") > 0;

			bool allFilters = filteredDetectors.count(setName) == 0;

            DetectorFilter detectorFilterProfile{};

            if (!allFilters) {

            }

            if (globalFilters) {
                allFilters = false;
            }

            addDetector("SIFT", cv::SIFT::create(), cv::NORM_L2);
            addDetector("ORB", ORB::create(250000), NORM_HAMMING);
            addDetector("BRISK", BRISK::create(), NORM_HAMMING);
            addDetector("SURF", xfeatures2d::SURF::create(), NORM_L2);

            std::vector<int> windowSizes = getWindowSize(reference.cols, reference.rows);

            std::cout << "  Using window sizes L = " << joinInts(windowSizes) << std::endl;

            if (allFilters || detectorFilterProfile.LPSIFT)
                addDetector("LP-SIFT", LPSIFT::create(
                    windowSizes
                ), NORM_L2);

            auto results = runAllDetectors(setName, reference, registered,
                windowSizes, outputPath);
            allResults.insert(allResults.end(), results.begin(), results.end());

        }

    }

    return allResults;
}

void BenchmarkRunner::printSummaryTable(const std::vector<StitchingMetrics>& results) {
    std::cout << "\n" << std::string(120, '=') << std::endl;
    std::cout << "BENCHMARK SUMMARY" << std::endl;
    std::cout << std::string(120, '=') << std::endl;

    // Header
    std::cout << std::left
              << std::setw(15) << "Dataset"
              << std::setw(10) << "Size"
              << std::setw(12) << "Algorithm"
              << std::setw(14) << "Resolution"
              << std::setw(12) << "Keypts Ref"
              << std::setw(12) << "Keypts Reg"
              << std::setw(10) << "Matches"
              << std::setw(10) << "Inliers"
              << std::setw(12) << "Window(L)"
              << std::setw(12) << "Time(s)"
              << std::setw(48) << "Homography Difference"
              << std::endl;
    std::cout << std::string(120, '-') << std::endl;

    for (const auto& m : results) {
        std::cout << std::left
                  << std::setw(15) << m.datasetName.substr(0, 14)
                  << std::setw(10) << imageSizeCategoryToString(m.sizeCategory)
                  << std::setw(12) << m.algorithmName
                  << std::setw(14) << m.getReferenceResolution()
                  << std::setw(12) << (m.stitchingSuccess ? std::to_string(m.numKeypointsReference) : "x")
                  << std::setw(12) << (m.stitchingSuccess ? std::to_string(m.numKeypointsRegistered) : "x")
                  << std::setw(10) << (m.stitchingSuccess ? std::to_string(m.numMatches) : "x")
                  << std::setw(10) << (m.stitchingSuccess ? std::to_string(m.numInliers) : "x")
                  << std::setw(12) << m.windowSizes
                  << std::setw(12) << (m.stitchingSuccess ? StitchingMetrics::formatTime(m.totalStitchingTime) : "Failed")
                  << std::setw(48) << m.printHomography(m.homography - m.baselineH)
                  << std::endl;
    }

    std::cout << std::string(120, '=') << std::endl;
}

cv::Mat BenchmarkRunner::warpAndBlend(const cv::Mat& imgToWarp, const cv::Mat& baseImg, const cv::Mat& H) {
    // Calculate corners of images
    std::vector<cv::Point2f> cornersWarp = {
        cv::Point2f(0, 0),
        cv::Point2f(static_cast<float>(imgToWarp.cols), 0),
        cv::Point2f(static_cast<float>(imgToWarp.cols), static_cast<float>(imgToWarp.rows)),
        cv::Point2f(0, static_cast<float>(imgToWarp.rows))
    };

    std::vector<cv::Point2f> cornersBase = {
        cv::Point2f(0, 0),
        cv::Point2f(static_cast<float>(baseImg.cols), 0),
        cv::Point2f(static_cast<float>(baseImg.cols), static_cast<float>(baseImg.rows)),
        cv::Point2f(0, static_cast<float>(baseImg.rows))
    };

	float minX = FLT_MAX, minY = FLT_MAX;
	float maxX = -FLT_MAX, maxY = -FLT_MAX;

    std::vector<cv::Point2f> warpedCorners;
    cv::perspectiveTransform(cornersWarp, warpedCorners, H);

    for (const auto& p : warpedCorners) {
        minX = std::min(minX, p.x);
        minY = std::min(minY, p.y);
        maxX = std::max(maxX, p.x);
        maxY = std::max(maxY, p.y);
	}

    for (const auto& p : cornersBase) {
        minX = std::min(minX, p.x);
        minY = std::min(minY, p.y);
        maxX = std::max(maxX, p.x);
        maxY = std::max(maxY, p.y);
    }

    int offsetX = (minX < 0) ? static_cast<int>(-minX) : 0;
    int offsetY = (minY < 0) ? static_cast<int>(-minY) : 0;

    int width = static_cast<int>(maxX - minX + 1);
    int height = static_cast<int>(maxY - minY + 1);
    
    // Warp and blend
    cv::Mat stitched = cv::Mat::zeros(cv::Size(width, height), baseImg.type());

    // Create translation matrix
    cv::Mat T = (cv::Mat_<double>(3, 3) <<
        1, 0, offsetX,
        0, 1, offsetY,
        0, 0, 1);

    cv::Mat Hshifted = T * H;

    // First warp the secondary image, then place the reference on top
    cv::warpPerspective(imgToWarp, stitched, Hshifted, cv::Size(width, height),
        cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    cv::Rect roi(offsetX, offsetY, baseImg.cols, baseImg.rows);
    baseImg.copyTo(stitched(roi));

    return stitched;
}
