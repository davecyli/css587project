/*
 * David Li, Ben Schipunov, Kris Yu
 * 11/14/2025
 * CSS 587
 * Final Project: LP-SIFT
 *
 * main.cpp
 * Main driver file for the program
 *
 * Features included:
 * - Image stitching with multiple feature detectors
 * - Benchmarking framework with CSV export
 * - Performance metrics collection (timing, keypoints, matches, etc.)
 *
 * Assumptions and constraints:
 * - C++17 is used
 *
 * Usage:
 *   ./css587project                      - Run visual demo on all image sets
 *   ./css587project --benchmark          - Run full benchmark, output to results.csv
 *   ./css587project --benchmark -o out.csv  - Run benchmark with custom output file
 *   ./css587project --help               - Show help
 *   ./css587project <set1> <set2> ...    - Run demo on specific image sets
 */

#include <string>
#include <stdexcept>
#include <iostream>
#include <cstddef>
#include <vector>
#include <filesystem>
#include <set>
#include <map>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>  // Uncomment for SURF support
#include <opencv2/imgproc.hpp>

#include "lpsift.h"
#include "benchmark.h"

using namespace std;
using namespace cv;

namespace fs = std::filesystem;

const string IMAGE_DIR = "images";
const string DEFAULT_OUTPUT_CSV = "results.csv";

int WINDOW_WIDTH = 800;
int WINDOW_HEIGHT = 600;

// Folder Structure
// images/
//    set1/ [any name]
//       registered.jpg
//       reference.jpg
//    anotherset2/
//       registered.jpg
//       reference.jpg
//   ...

void printUsage(const char* programName) {
	cout << "CSS 587 Final Project: LP-SIFT Implementation and Benchmarking\n"
	     << "David Li, Ben Schipunov, Kris Yu\n\n"
	     << "Usage:\n"
	     << "  " << programName << "                           Run visual demo on all image sets\n"
	     << "  " << programName << " --benchmark               Run full benchmark, save to results.csv\n"
	     << "  " << programName << " --benchmark -o <file>     Run benchmark with custom output file\n"
	     << "  " << programName << " --benchmark --save-images Save stitched images during benchmark\n"
	     << "  " << programName << " <set1> <set2> ...         Run demo on specific image sets\n"
	     << "  " << programName << " --help                    Show this help message\n\n"
	     << "Options:\n"
	     << "  --benchmark       Run performance benchmarks instead of visual demo\n"
	     << "  -o, --output      Specify output CSV file (default: results.csv)\n"
	     << "  --save-images     Save stitched images during benchmark\n"
	     << "  --no-display      Skip visual display in demo mode\n"
	     << endl;
}

// Helper function to limit keypoints by keeping the strongest ones
// Uses MAX_KEYPOINTS constant from benchmark.h
void limitKeypoints(vector<KeyPoint>& kpts, int maxCount) {
	if (static_cast<int>(kpts.size()) > maxCount) {
		// Sort by response (strength) descending and keep top N
		sort(kpts.begin(), kpts.end(),
			[](const KeyPoint& a, const KeyPoint& b) {
				return a.response > b.response;
			});
		kpts.resize(maxCount);
	}
}

bool runCase(string set_name, const Mat& img1, const Mat& img2, const Ptr<Feature2D>& detector, NormTypes matcher_norm) {
	Mat gray1, gray2;

	cvtColor(img1, gray1, COLOR_BGR2GRAY);
	cvtColor(img2, gray2, COLOR_BGR2GRAY);

	vector<KeyPoint> kpts1, kpts2;
	Mat desc1, desc2;

	// Detect keypoints first
	detector->detect(gray1, kpts1);
	detector->detect(gray2, kpts2);

	cout << "Method: " << detector->getDefaultName()
	     << ", Keypoints detected - Image 1: " << kpts1.size()
	     << ", Image 2: " << kpts2.size();

	// Limit keypoints to prevent BFMatcher overflow
	limitKeypoints(kpts1, MAX_KEYPOINTS);
	limitKeypoints(kpts2, MAX_KEYPOINTS);

	if (kpts1.size() < static_cast<size_t>(MAX_KEYPOINTS) || kpts2.size() < static_cast<size_t>(MAX_KEYPOINTS)) {
		// Only print if we actually limited
	} else {
		cout << " (limited to " << MAX_KEYPOINTS << " each)";
	}
	cout << endl;

	// Compute descriptors for the (possibly limited) keypoints
	detector->compute(gray1, kpts1, desc1);
	detector->compute(gray2, kpts2, desc2);

	cout << "  After compute - Keypoints Image 1: " << kpts1.size()
	     << ", Image 2: " << kpts2.size() << endl;

	if (kpts1.empty() || kpts2.empty() || desc1.empty() || desc2.empty()) {
		cout << "Skipping " << detector->getDefaultName() << " due to empty keypoints/descriptors." << endl;
		return false;
	}

	Ptr<BFMatcher> matcher = BFMatcher::create(matcher_norm);
	vector<DMatch> matches;
	matcher->match(desc1, desc2, matches);

	if (matches.size() < 4) {
		cout << "Skipping " << detector->getDefaultName() << " due to insufficient matches." << endl;
		return false;
	}

	// Get matched points
	vector<Point2f> pts1, pts2;
	for (auto& m : matches) {
		pts1.push_back(kpts1[m.queryIdx].pt);
		pts2.push_back(kpts2[m.trainIdx].pt);
	}

	setRNGSeed(RNG_SEED); // for reproducibility (constant from benchmark.h)

	// Compute homography
	Mat H = findHomography(pts1, pts2, RANSAC);

	vector<Point2f> corners1 = {
		Point2f(0,0),
		Point2f(static_cast<float>(img1.cols),0),
		Point2f(static_cast<float>(img1.cols),static_cast<float>(img1.rows)),
		Point2f(0,static_cast<float>(img1.rows))
	};

	vector<Point2f> corners2 = {
		Point2f(0,0),
		Point2f(static_cast<float>(img2.cols),0),
		Point2f(static_cast<float>(img2.cols),static_cast<float>(img2.rows)),
		Point2f(0,static_cast<float>(img2.rows))
	};

	float minX = FLT_MAX, minY = FLT_MAX;
	float maxX = -FLT_MAX, maxY = -FLT_MAX;

	vector<Point2f> warpedCorners2;
	perspectiveTransform(corners2, warpedCorners2, H);

	for (auto& p : corners1) {
		minX = std::min(minX, p.x);
		minY = std::min(minY, p.y);
		maxX = std::max(maxX, p.x);
		maxY = std::max(maxY, p.y);
	}

	// Get overall min and max bounds of the resultant stitched image
	for (auto& p : warpedCorners2)
	{
		minX = std::min(minX, p.x);
		minY = std::min(minY, p.y);
		maxX = std::max(maxX, p.x);
		maxY = std::max(maxY, p.y);
	}

	int offsetX = (minX < 0) ? static_cast<int>(-minX) : 0;
	int offsetY = (minY < 0) ? static_cast<int>(-minY) : 0;

	int width = static_cast<int>(maxX - minX + 1);
	int height = static_cast<int>(maxY - minY + 1);

	Mat stitched = Mat::zeros(Size(width, height), img1.type());

	Mat T = (Mat_<double>(3, 3) << // Shift homography
		1, 0, offsetX,
		0, 1, offsetY,
		0, 0, 1);

	Mat Hshifted = T * H;

	warpPerspective(img1, stitched, Hshifted, Size(width, height));

	// Safe ROI calculation
	int roiWidth = std::min(img2.cols, width - offsetX);
	int roiHeight = std::min(img2.rows, height - offsetY);
	if (roiWidth > 0 && roiHeight > 0) {
		Mat roi(stitched, Rect(offsetX, offsetY, roiWidth, roiHeight));
		img2(Rect(0, 0, roiWidth, roiHeight)).copyTo(roi);
	}

	string windowName = "Stitched - " + set_name + " - " + detector->getDefaultName();

	// Get scale to fit desired window size
	double scale = max(1.0, min(stitched.cols / (double)WINDOW_WIDTH, stitched.rows / (double)WINDOW_HEIGHT));

	namedWindow(windowName, WINDOW_NORMAL);
	resizeWindow(windowName, (int)(stitched.cols / scale), (int)(stitched.rows / scale));
	imshow(windowName, stitched);

	return true;
}

// Run benchmark mode
int runBenchmark(const string& outputFile, bool saveImages) {
	cout << "=================================================\n"
	     << "CSS 587 LP-SIFT Benchmarking Framework\n"
	     << "=================================================\n\n";

	if (!fs::exists(IMAGE_DIR) || !fs::is_directory(IMAGE_DIR)) {
		cerr << "Error: Image directory does not exist: " << IMAGE_DIR << endl;
		return 1;
	}

	BenchmarkRunner runner;

	// Add all detectors to benchmark (matching paper's Table 2)
	// SIFT - uses L2 norm for matching
	runner.addDetector("SIFT", SIFT::create(), NORM_L2, "x");

	// ORB - uses Hamming distance
	runner.addDetector("ORB", ORB::create(), NORM_HAMMING, "x");

	// BRISK - uses Hamming distance
	runner.addDetector("BRISK", BRISK::create(), NORM_HAMMING, "x");

	// SURF - uncomment if xfeatures2d is available
	runner.addDetector("SURF", xfeatures2d::SURF::create(), NORM_L2, "x");

	// LP-SIFT - uses L2 norm (like SIFT) with window sizes
	// Note: LP-SIFT detection is being implemented by teammate
	// When ready, uncomment:
	//runner.addDetector("LP-SIFT", LPSIFT::create(), NORM_L2, "[32,64]");

	cout << "Image directory: " << IMAGE_DIR << endl;
	cout << "Output file: " << outputFile << endl;
	cout << "Save stitched images: " << (saveImages ? "Yes" : "No") << endl;
	cout << "\nStarting benchmark...\n" << endl;

	// Create output directory for stitched images if needed
	string outputDir = "benchmark_output";
	if (saveImages) {
		fs::create_directories(outputDir);
	}

	// Run benchmarks on all image sets
	auto results = runner.runOnDirectory(IMAGE_DIR, saveImages, outputDir);

	if (results.empty()) {
		cerr << "No benchmark results collected. Check if images exist in " << IMAGE_DIR << endl;
		return 1;
	}

	// Export to CSV
	CSVExporter exporter(outputFile);
	exporter.writeAllMetrics(results);
	cout << "\nResults saved to: " << outputFile << endl;

	// Print summary table
	BenchmarkRunner::printSummaryTable(results);

	// Print statistics summary
	cout << "\nStatistics by Algorithm:" << endl;
	cout << string(60, '-') << endl;

	map<string, vector<double>> timesByAlgo;
	map<string, int> successCount;
	map<string, int> totalCount;

	for (const auto& m : results) {
		totalCount[m.algorithmName]++;
		if (m.stitchingSuccess) {
			timesByAlgo[m.algorithmName].push_back(m.totalStitchingTime);
			successCount[m.algorithmName]++;
		}
	}

	for (const auto& [algo, times] : timesByAlgo) {
		if (times.empty()) continue;

		double sum = 0;
		double minTime = times[0];
		double maxTime = times[0];
		for (double t : times) {
			sum += t;
			minTime = std::min(minTime, t);
			maxTime = std::max(maxTime, t);
		}
		double avg = sum / times.size();

		cout << algo << ":\n"
		     << "  Success rate: " << successCount[algo] << "/" << totalCount[algo] << "\n"
		     << "  Avg time: " << StitchingMetrics::formatTime(avg) << "s\n"
		     << "  Min time: " << StitchingMetrics::formatTime(minTime) << "s\n"
		     << "  Max time: " << StitchingMetrics::formatTime(maxTime) << "s\n"
		     << endl;
	}

	return 0;
}

// Run visual demo mode (original behavior)
int runDemo(const set<string>& filteredImageIds, bool showDisplay) {
	if (!fs::exists(IMAGE_DIR) || !fs::is_directory(IMAGE_DIR)) {
		throw runtime_error("Image directory does not exist: " + IMAGE_DIR);
	}

	vector<string> image_set_paths;

	for (const auto& entry : fs::directory_iterator(IMAGE_DIR)) {
		if (entry.is_directory()) {
			image_set_paths.push_back(entry.path().string());
		}
	}

	sort(image_set_paths.begin(), image_set_paths.end());

	bool anyDisplayed = false;

	for (const string& image_set_path : image_set_paths) {
		// Handle both forward and back slashes for cross-platform compatibility
		size_t last_slash_index = image_set_path.find_last_of("/\\");
		string dir_name = image_set_path.substr(last_slash_index == string::npos ? 0 : last_slash_index + 1);

		// C++17: std::set::contains is C++20. Use find(...) != end() instead.
		if (filteredImageIds.empty() || filteredImageIds.find(dir_name) != filteredImageIds.end()) {

			cout << "Processing image set: " << dir_name << endl;

			cv::Mat imgRegistered = cv::imread(image_set_path + "/registered.jpg");
			cv::Mat imgReference = cv::imread(image_set_path + "/reference.jpg");

			if (imgRegistered.empty() || imgReference.empty()) {
				cerr << "Warning: Could not load images from " << image_set_path << endl;
				continue;
			}

			// Define detectors to test
			// Uncomment/comment as needed
			pair<Ptr<Feature2D>, NormTypes> detectors[] = {
				{SIFT::create(), NORM_L2},
				{ORB::create(), NORM_HAMMING},
				{BRISK::create(), NORM_HAMMING},
				{xfeatures2d::SURF::create(), NORM_L2},  // Uncomment if xfeatures2d available
				//{LPSIFT::create(), NORM_L2}  // Uncomment when LP-SIFT detection is implemented
			};

			for (auto& detectorEntry : detectors) {
				Ptr<Feature2D> detector = detectorEntry.first;
				NormTypes norm_type = detectorEntry.second;
				bool shown = runCase(dir_name, imgRegistered, imgReference, detector, norm_type);
				anyDisplayed = anyDisplayed || shown;
			}
		}
	}

	if (showDisplay && anyDisplayed) {
		cout << "\nPress any key to close windows..." << endl;
		waitKey(0);
	}

	return 0;
}

int main(int argc, char* argv[]) {
	// Parse command line arguments
	bool benchmarkMode = false;
	bool saveImages = false;
	bool showDisplay = true;
	string outputFile = DEFAULT_OUTPUT_CSV;
	set<string> filteredImageIds;

	for (int i = 1; i < argc; i++) {
		string arg = argv[i];

		if (arg == "--help" || arg == "-h") {
			printUsage(argv[0]);
			return 0;
		}
		else if (arg == "--benchmark" || arg == "-b") {
			benchmarkMode = true;
		}
		else if (arg == "--output" || arg == "-o") {
			if (i + 1 < argc) {
				outputFile = argv[++i];
			} else {
				cerr << "Error: --output requires a filename argument" << endl;
				return 1;
			}
		}
		else if (arg == "--save-images") {
			saveImages = true;
		}
		else if (arg == "--no-display") {
			showDisplay = false;
		}
		else if (arg[0] != '-') {
			// Assume it's an image set filter
			filteredImageIds.insert(arg);
		}
		else {
			cerr << "Unknown option: " << arg << endl;
			printUsage(argv[0]);
			return 1;
		}
	}

	try {
		if (benchmarkMode) {
			return runBenchmark(outputFile, saveImages);
		} else {
			return runDemo(filteredImageIds, showDisplay);
		}
	}
	catch (const exception& e) {
		cerr << "Error: " << e.what() << endl;
		return 1;
	}
}
