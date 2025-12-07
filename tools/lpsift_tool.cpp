/*
 * Simple LP-SIFT driver adapted from css587-p3 homework driver.
 * Loads two images, runs LP-SIFT detect+compute, filters matches with RANSAC,
 * prints a summary, and optionally draws/saves the match visualization.
 */

#include "lpsift.h"

#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;

const std::string kDefaultOut = "lpsift_matches.jpg";
const int kColWidth = 8;
const int kScale = 6;
const double kRansacThresh = 3.0;

// -----------------------------------------------------------------------------
// loadImageOrFail - Loads an image from disk or logs an error if loading fails.
// preconditions:
//   - 'path' references a readable image file on disk.
// postconditions:
//   - Returns a CV_8UC3 Mat if successful; otherwise returns an empty Mat and
//     prints an error to std::cerr.
static Mat loadImageOrFail(const std::string& path) {
    Mat img = imread(path, IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to read: " << path << "\n";
    }
    return img;
}

// -----------------------------------------------------------------------------
// parseWindowSizes - Parses comma-separated ints, keeps positives only.
// preconditions:
//   - 'arg' may be empty; tokens convertible via std::stoi.
// postconditions:
//   - Returns a non-empty vector, falling back to LP-SIFT defaults when none
//     are parsed.
static std::vector<int> parseWindowSizes(const std::string& arg) {
    std::vector<int> result;
    std::stringstream ss(arg);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try {
            const int v = std::stoi(token);
            if (v > 0) result.push_back(v);
        } catch (...) {
            // Ignore parse failures and continue.
        }
    }
    if (result.empty()) result = {16, 40, 128};
    return result;
}

// -----------------------------------------------------------------------------
// runAndShow - Runs the specified detector, filters using RANSAC, shows matches,
//              and logs timings for a single detector. (optionally saves)
// preconditions:
//   - 'img1' and 'img2' are non-empty CV_8UC3 Mats of the same scene.
//   - 'detector' is a valid cv::Feature2D (supports detectAndCompute).
//   - 'normType' matches descriptor type.
//   - If 'savePath' is non-empty, the process has permission to write there.
// postconditions:
//   - If 'savePath' is non-empty, writes the visualization image to disk.
//   - Prints a single summary line containing counts and timing (ms) for
//     detect+compute, matching and keypoints/ms.
//   - Optionally shows a resized window if 'showOutput' is true.
//   - Returns immediately after a short non-blocking waitKey(1).
static void runAndShow(const Mat& img1,
                       const Mat& img2,
                       Ptr<Feature2D> detector,
                       int normType,
                       const std::string& label,
                       const std::string& savePath,
                       bool showOutput) {
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    TickMeter tm; // For timing metrics

    // Detect and compute keypoints and descriptors
    tm.start();
    detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);
    tm.stop();
    const double detect_ms = tm.getTimeMilli();

    Ptr<BFMatcher> matcher = BFMatcher::create(normType, true); // crossCheck=true
    std::vector<DMatch> matches;

    // Match descriptors
    tm.reset();
    tm.start();
    matcher->match(descriptors1, descriptors2, matches);
    tm.stop();
    const double match_ms = tm.getTimeMilli();

    // RANSAC inlier filtering
    std::vector<Point2f> srcPoints, dstPoints;
    srcPoints.reserve(matches.size());
    dstPoints.reserve(matches.size());
    for (const auto& m : matches) {
        srcPoints.push_back(keypoints1[m.queryIdx].pt);
        dstPoints.push_back(keypoints2[m.trainIdx].pt);
    }

    std::vector<uchar> inlierMask;
    if (matches.size() >= 4) {
        findHomography(srcPoints, dstPoints, RANSAC, kRansacThresh, inlierMask);
    }

    // Filter matches using the mask (or keep all if homography failed)
    std::vector<DMatch> inlierMatches;
    if (!inlierMask.empty()) {
        for (size_t i = 0; i < inlierMask.size(); ++i) {
            if (inlierMask[i]) inlierMatches.push_back(matches[i]);
        }
    } else {
        inlierMatches = matches;
    }
    const double inlierPct = matches.empty()
                                 ? 0.0
                                 : 100.0 * static_cast<double>(inlierMatches.size()) /
                                       static_cast<double>(matches.size());

    // Draw inliers (or all matches if homography failed)
    Mat output;
    drawMatches(img1, keypoints1, img2, keypoints2, inlierMatches, output);

    // Save output image if path supplied
    if (!savePath.empty()) imwrite(savePath, output);

    // Display output image in resized window
    if (showOutput) {
        namedWindow(label, WINDOW_NORMAL);
        resizeWindow(label, output.cols / kScale, output.rows / kScale);
        imshow(label, output);
        waitKey(1); // ensure each window renders before moving on
    }

    // Calculate derived metrics and print summary line
    const double total_kp = static_cast<double>(keypoints1.size() + keypoints2.size());
    const double kp_per_ms = detect_ms > 0.0 ? total_kp / detect_ms : 0.0;

    std::cout << std::left << std::setw(kColWidth) << label << " |"
              << " kp1: " << std::setw(kColWidth) << keypoints1.size()
              << " kp2: " << std::setw(kColWidth) << keypoints2.size()
              << " matches: " << std::setw(kColWidth) << matches.size()
              << " inlier(%): " << std::setw(kColWidth) << std::fixed << std::setprecision(2)
              << inlierPct
              << " detect(ms): " << std::setw(kColWidth) << std::fixed << std::setprecision(1)
              << detect_ms
              << " match(ms): " << std::setw(kColWidth) << std::fixed << std::setprecision(1)
              << match_ms
              << " kp/ms: " << std::setw(kColWidth) << std::fixed << std::setprecision(2)
              << kp_per_ms
              << (savePath.empty() ? "" : " -> saved " + savePath) << "\n";
}

// --------------------------------- Main --------------------------------------
// main - Program entry. Runs LP-SIFT on two images and writes matches.
// preconditions:
//   - First two args are readable image paths.
//   - OpenCV core/highgui/features2d/imgcodecs are available at runtime.
// postconditions:
//   - Writes a visualization image if an output path is provided.
//   - Prints a performance summary line for LP-SIFT.
//   - Optionally shows a window with match visualization.
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: lpsift_tool <image1> <image2> [output_path] [window_sizes]\n"
                     "  window_sizes: comma-separated (e.g., 16,40,128); defaults to LP-SIFT defaults\n"
                     "  Set output_path to '' to skip saving; visualization window is shown by default.\n";
        return 1;
    }

    const std::string img1Path = argv[1];
    const std::string img2Path = argv[2];
    const std::string outPath = argc >= 4 ? argv[3] : kDefaultOut;
    const std::vector<int> windowSizes = argc >= 5 ? parseWindowSizes(argv[4]) : std::vector<int>{40, 128};
    const bool showOutput = true;

    Mat img1 = loadImageOrFail(img1Path);
    Mat img2 = loadImageOrFail(img2Path);
    if (img1.empty() || img2.empty()) return 1;

    Ptr<LPSIFT> lpsift = LPSIFT::create(windowSizes, 1e-6f);
    runAndShow(img1, img2, lpsift, NORM_HAMMING, "LP-SIFT", outPath, showOutput);

    return 0;
}
