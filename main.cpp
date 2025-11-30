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
 * - C++17 is used
 */

#include <string>
#include <stdexcept>
#include <iostream>
#include <cstddef>
#include <vector>
#include <filesystem>
#include <set>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
 //#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>

#include "lpsift.h"

using namespace std;
using namespace cv;

namespace fs = std::filesystem;

const string IMAGE_DIR = "images";

int WINDOW_WIDTH = 800;
int WINDOW_HEIGHT = 600;

// Folder Structure
// images/
//    set1/ [any name]
//       img1.jpg [these can be any name, and can index arbitrarily]
//       img2.jpg [we can assume only 2 images per folder with our two-image tests]
//    anotherset2/
//       someimage.jpg
//       partofthisimage.jpg
//   ...

int runCase(string set_name, const Mat& img1, const Mat& img2, const string& method_name, const Ptr<Feature2D>& detector, NormTypes matcher_norm) {

	Mat gray1, gray2;

	cvtColor(img1, gray1, COLOR_BGR2GRAY);
	cvtColor(img2, gray2, COLOR_BGR2GRAY);

	vector<KeyPoint> kpts1, kpts2;
	Mat desc1, desc2;

	detector->detectAndCompute(gray1, noArray(), kpts1, desc1);
	detector->detectAndCompute(gray2, noArray(), kpts2, desc2);

	cout << "Method: " << method_name << ", Keypoints Image 1: " << kpts1.size() << ", Keypoints Image 2: " << kpts2.size() << endl;

	Ptr<BFMatcher> matcher = BFMatcher::create(matcher_norm);
	vector<DMatch> matches;
	matcher->match(desc1, desc2, matches);

	// Get matched points
	vector<Point2f> pts1, pts2;
	for (auto& m : matches) {
		pts1.push_back(kpts1[m.queryIdx].pt);
		pts2.push_back(kpts2[m.trainIdx].pt);
	}

	setRNGSeed(12345); // for reproducibility

	// Compute homography
	Mat H = findHomography(pts1, pts2, RANSAC);

	vector<Point2f> corners1 = {
		Point2f(0,0),
		Point2f(img1.cols,0),
		Point2f(img1.cols,img1.rows),
		Point2f(0,img1.rows)
	};

	vector<Point2f> corners2 = {
		Point2f(0,0),
		Point2f(img2.cols,0),
		Point2f(img2.cols,img2.rows),
		Point2f(0,img2.rows)
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

	int offsetX = (minX < 0) ? -minX : 0;
	int offsetY = (minY < 0) ? -minY : 0;

	int width = (int)(maxX - minX + 1);
	int height = (int)(maxY - minY + 1);

	Mat stitched = Mat::zeros(Size(width, height), img1.type());

	Mat T = (Mat_<double>(3, 3) << // Shift homography
		1, 0, offsetX,
		0, 1, offsetY,
		0, 0, 1);

	Mat Hshifted = T * H;

	warpPerspective(img1, stitched, Hshifted, Size(width, height));

	Mat roi(stitched, Rect(offsetX, offsetY, max(Hshifted.cols, img2.cols), max(Hshifted.rows, img2.rows)));
	img2.copyTo(roi);

	string windowName = "Stitched - " + set_name + " - " + method_name;

	// Get scale to fit desired window size
	double scale = max(1.0, min(stitched.cols / (double)WINDOW_WIDTH, stitched.rows / (double)WINDOW_HEIGHT));

	namedWindow(windowName, WINDOW_NORMAL);
	resizeWindow(windowName, (int)(stitched.cols / scale), (int)(stitched.rows / scale));
	imshow(windowName, stitched);

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
			string dir_name = image_set_path.substr(last_slash_index == string::npos ? 0 : last_slash_index);

			// C++17: std::set::contains is C++20. Use find(...) != end() instead.
			if (filtered_image_ids.empty() || filtered_image_ids.find(dir_name) != filtered_image_ids.end()) {

				cout << "Processing image set: " << dir_name << endl;

				cv::Mat imgRegistered = cv::imread(image_set_path + "/registered.jpg");
				cv::Mat imgReference = cv::imread(image_set_path + "/reference.jpg");

				pair<Ptr<Feature2D>, NormTypes> detectors[] = {
					//{SIFT::create(), NORM_L2},
					{ORB::create(), NORM_HAMMING},
					//{BRISK::create(), NORM_HAMMING},
					//{SURF::create(), NORM_HAMMING}, // in xfeatures2d
					//{LPSIFT::create(), NORM_L2}
				};

				for (auto detectorEntry : detectors) {

					Ptr<Feature2D> detector = detectorEntry.first;
					NormTypes norm_type = detectorEntry.second;

					string method_name = typeid(*detector).name();
					runCase(dir_name, imgRegistered, imgReference, method_name, detector, norm_type);
				}
			}

		}

		waitKey(0);
	}
	else {
		throw runtime_error("Image directory does not exist: " + IMAGE_DIR);
	}

	return 0;

}