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

#include "lpsift.h"

using namespace std;

namespace fs = std::filesystem;

const string IMAGE_DIR = "images";

// Folder Structure
// images/
//    pair1/
//       img1.jpg [these can be any name, and can index arbitrarily]
//       img2.jpg [we can assume only 2 images per folder with our two-image tests]
//    pair2/
//       someimage.jpg
//       partofthisimage.jpg
//   ...

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
		for (const auto& entry : fs::directory_iterator(IMAGE_DIR)) {
			if (entry.is_directory()) {
				
				string dir_name = entry.path().filename().string();
				
				if (filtered_image_ids.contains(dir_name) || filtered_image_ids.empty()) {
					
					cout << "Processing image set: " << dir_name << endl;

					vector<string> image_paths;

					for (const auto& image : fs::directory_iterator(entry.path())) {
						image_paths.push_back(image.path().string());
					}

					sort(image_paths.begin(), image_paths.end()); // ensure consistency in ordering

					for (int i = 0; i < image_paths.size(); i++) {
						cout << " - Image [" << i << "]: " << image_paths[i] << endl;
					}

					if (image_paths.size() != 2) {
						cerr << "Warning: Expected 2 images in directory " << dir_name << ", found " << image_paths.size() << ". Skipping this set." << endl;
						continue;
					}

					cout << " - Loading images..." << endl;

					cv::Mat img1 = cv::imread(image_paths[0], cv::IMREAD_GRAYSCALE);
					cv::Mat img2 = cv::imread(image_paths[1], cv::IMREAD_GRAYSCALE);

					cout << " - Running LP-SIFT on them..." << endl;

					// run LP-SIFT on this image pair

				}

			}
		}
	}
	else {
		throw runtime_error("Image directory does not exist: " + IMAGE_DIR);
	}

	return 0;

}
