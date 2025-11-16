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

using namespace std;

const string IMAGE_PAIRS[] = {

	"college",
	"mountains"
	// add more when needed
};

string getImageId(const string& token) {
	try {
		int index = stoi(token);
		if (index >= 0 && index < sizeof(IMAGE_PAIRS) / sizeof(IMAGE_PAIRS[0])) {
			return IMAGE_PAIRS[index];
		}
		throw out_of_range("Index " + to_string(index) + " is out of range [0, " +
		                    to_string(sizeof(IMAGE_PAIRS) / sizeof(IMAGE_PAIRS[0]) - 1) + "]");
	} catch (const invalid_argument&) {
		// Treat as name - could add validation here
		return token;
	}
};

void runCase(const string& image_pair) {
	cout << "Processing image pair: " << image_pair << endl;
	// TODO: Implement LP-SIFT algorithm for the image pair
}

int main(int argc, char* argv[]) {

	// Argument Usage:
	// default no args: run all cases
	// arg1 = image sets to run (comma-delimited) (they can be by name or index values)
	// examples: "college,mountains" or "0,1"

	if (argc == 1) { // no args, run all cases
		for (const string& image_pair : IMAGE_PAIRS) {
			runCase(image_pair);
		}
	} else { // parse args
		string arg1 = argv[1];
		size_t start = 0;
		size_t end = arg1.find(',');
		while (end != string::npos) {
			
			string token = arg1.substr(start, end - start);

			string imageId = getImageId(token);

			runCase(imageId);
			start = end + 1;
			end = arg1.find(',', start);
		}

		string token = arg1.substr(start);

		string imageId = getImageId(token);
		
		runCase(imageId);

	}

	return 0;

}
