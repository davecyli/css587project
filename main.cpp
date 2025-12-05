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
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>

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
		<< "  " << programName << " <set1> <set2> ...         Run demo on specific image sets\n"
		<< "  " << programName << " --help                    Show this help message\n\n"
		<< endl;
}

// Helper function to limit keypoints by keeping the strongest ones
// Uses MAX_KEYPOINTS_BF constant from benchmark.h for BFMatcher limit
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

// Run benchmark mode
int runBenchmark(const set<string>& filteredImageSets) {
	cout << "=================================================\n"
		<< "CSS 587 LP-SIFT Benchmarking Framework\n"
		<< "=================================================\n\n";

	if (!fs::exists(IMAGE_DIR) || !fs::is_directory(IMAGE_DIR)) {
		cerr << "Error: Image directory does not exist: " << IMAGE_DIR << endl;
		return 1;
	}

	BenchmarkRunner runner;

	cout << "Image directory: " << IMAGE_DIR << endl;
	cout << "\nStarting benchmark...\n" << endl;

	// Create output directory for stitched images if needed
	string outputDir = "benchmark_output";
	fs::create_directories(outputDir);

	// Run benchmarks on all image sets
	auto results = runner.runOnDirectory(IMAGE_DIR, filteredImageSets, outputDir);

	if (results.empty()) {
		cerr << "No benchmark results collected. Check if images exist in " << IMAGE_DIR << endl;
		return 1;
	}

	std::string outputFile = DEFAULT_OUTPUT_CSV;

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

int main(int argc, char* argv[]) {

	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

	// Parse command line arguments
	string outputFile = DEFAULT_OUTPUT_CSV;
	set<string> filteredImageIds;

	for (int i = 1; i < argc; i++) {
		string arg = argv[i];

		if (arg == "--help" || arg == "-h") {
			printUsage(argv[0]);
			return 0;
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
		return runBenchmark(filteredImageIds);
	}
	catch (const exception& e) {
		cerr << "Error: " << e.what() << endl;
		return 1;
	}
}
