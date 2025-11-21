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
 * 
 * Assumptions and constraints:
 */

#include <string>
#include <stdexcept>
#include <iostream>
#include <cstddef>
#include <vector>
#include <filesystem>
#include <set>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
//#include <opencv2/xfeatures2d.hpp>

// #include "lpsift.h" temp commented out until lpsift.h is implemented

using namespace std;
using namespace cv;

namespace fs = std::filesystem;

const string IMAGE_DIR = "images";

// Folder Structure
// images/
//    set1/ [any name]
//       img1.jpg [these can be any name, and can index arbitrarily]
//       img2.jpg [we can assume only 2 images per folder with our two-image tests]
//    anotherset2/
//       someimage.jpg
//       partofthisimage.jpg
//   ...

template <typename T> int computeMatches(const Mat& image1, const Mat& image2, Mat& matches_out, const Ptr<T>& detector) {
	
	vector<KeyPoint> keypoints1; // keypoint vectors
	vector<KeyPoint> keypoints2;

	Mat descriptors1; // descriptor matrices
	Mat descriptors2;

	detector->detectAndCompute(image1, noArray(), keypoints1, descriptors1); // detect and compute for image1 and image2
	detector->detectAndCompute(image2, noArray(), keypoints2, descriptors2);

	Ptr<BFMatcher> matcher = BFMatcher::create();

	vector<DMatch> matches;
	matcher->match(descriptors1, descriptors2, matches); // match descriptors

	//drawMatches(image1, keypoints1, image2, keypoints2, matches, matches_out); // draw matches on output image

	return 0;

}

int main(int argc, char* argv[]) {

	// Argument Usage:
	// default no args: run all cases
	// arg1 = filter by image set IDs (separated by spaces) (they can be by name or index values) (e.g., pair1 pair2 ...)
	// for expansion (like config for CUDA), we may want to use flags and comma-delimit the image ID filter

	set<string> filtered_image_ids;

	for (int i = 1; i < argc; i++) {
		string arg = argv[i];
		filtered_image_ids.insert(arg);
	}

	// Get all image pairs from the directory
	if (fs::exists(IMAGE_DIR) && fs::is_directory(IMAGE_DIR)) {

		vector<string> image_set_paths;

		for (const auto& entry : fs::directory_iterator(IMAGE_DIR)) {
			if (entry.is_directory()) {
				image_set_paths.push_back(entry.path().string());
			}
		}

		sort(image_set_paths.begin(), image_set_paths.end());

		for (string image_set_path : image_set_paths) {
			
			int last_slash_index = image_set_path.rfind('/');
			string dir_name = image_set_path.substr(last_slash_index==string::npos ? 0 : last_slash_index);

			if (filtered_image_ids.contains(dir_name) || filtered_image_ids.empty()) {

				cout << "Processing image set: " << dir_name << endl;

				cv::Mat imgRegistered = cv::imread(image_set_path + "/registered.jpg", cv::IMREAD_GRAYSCALE);
				cv::Mat imgReferenced = cv::imread(image_set_path + "/referenced.jpg", cv::IMREAD_GRAYSCALE);

				cout << " - Running SIFT..." << endl;
				// run SIFT
				Ptr<SIFT> detectorSIFT = SIFT::create();

				cout << " - Running ORB..." << endl;
				// run ORB
				Ptr<ORB> detectorORB = ORB::create();

				cout << " - Running BRISK..." << endl;
				// run BRISK
				Ptr<BRISK> detectorBRISK = BRISK::create();

				cout << " - Running SURF..." << endl;
				// run SURF
				// SURF is in xfeatures2d which may not be included in all OpenCV builds

				cout << " - Running LP-SIFT..." << endl;
				// run LP-SIFT
				//Ptr<LP_SIFT> detectorLPSIFT = LP_SIFT::create();

			}

		}
	}
	else {
		throw runtime_error("Image directory does not exist: " + IMAGE_DIR);
	}

	return 0;

}
