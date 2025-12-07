/*
 * 2D spatial FFT visualizer and LP-SIFT window-size suggester.
 * Computes the magnitude spectrum of an image, finds the strongest frequency
 * components, and suggests window sizes based on their spatial periods.
 */

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
namespace fs = std::filesystem;

// -----------------------------------------------------------------------------
// loadImageOrFail - Loads an image from disk or logs an error if loading fails.
// preconditions:
//   - 'path' references a readable image file on disk.
// postconditions:
//   - Returns a CV_8UC1 Mat if successful; otherwise an empty Mat and an error.
static Mat loadImageOrFail(const std::string& path) {
    Mat img = imread(path, IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Failed to read: " << path << "\n";
    }
    return img;
}

// -----------------------------------------------------------------------------
// fftShift - Moves the zero-frequency component to the center of the spectrum.
// preconditions:
//   - 'complex' is a 2-channel DFT result.
// postconditions:
//   - Quadrants are swapped in-place.
static void fftShift(Mat& complex) {
    const int cx = complex.cols / 2;
    const int cy = complex.rows / 2;

    Mat q0(complex, Rect(0, 0, cx, cy));   // Top-Left
    Mat q1(complex, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(complex, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(complex, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

// -----------------------------------------------------------------------------
// findTopPeaks - Finds top-K peaks in the magnitude image, suppressing neighbors.
// preconditions:
//   - 'mag' is single-channel float magnitude spectrum (centered).
// postconditions:
//   - Returns up to topK peaks with their magnitude and location.
struct Peak {
    Point loc;
    double mag;
};

static std::vector<Peak> findTopPeaks(Mat mag, int topK, int suppressRadius) {
    std::vector<Peak> peaks;
    for (int i = 0; i < topK; ++i) {
        double maxVal = 0.0;
        Point maxLoc;
        minMaxLoc(mag, nullptr, &maxVal, nullptr, &maxLoc);
        if (maxVal <= 0) break;
        peaks.push_back({maxLoc, maxVal});
        circle(mag, maxLoc, suppressRadius, Scalar(0), -1);
    }
    return peaks;
}

// -----------------------------------------------------------------------------
// suggestWindow - Converts frequency bin offset to a spatial window size.
// preconditions:
//   - 'dx','dy' are offsets from DC (centered spectrum).
// postconditions:
//   - Returns an integer window size >= 2, bounded by min(rows, cols).
static int suggestWindow(int dx, int dy, int rows, int cols) {
    const double radius = std::sqrt(static_cast<double>(dx * dx + dy * dy));
    if (radius <= 0.0) return std::min(rows, cols);
    const double period = static_cast<double>(std::min(rows, cols)) / radius;
    const int L = static_cast<int>(std::round(std::clamp(period, 2.0, static_cast<double>(std::min(rows, cols)))));
    return L;
}

// -----------------------------------------------------------------------------
// makeSpectrumViz - Converts log-magnitude to displayable BGR with markers.
// preconditions:
//   - 'logMag' is log(magnitude + 1), float.
// postconditions:
//   - Returns an 8-bit BGR visualization with peak markers and labels.
static Mat makeSpectrumViz(const Mat& logMag,
                           const std::vector<Peak>& peaks,
                           const std::vector<int>& windows) {
    Mat norm;
    normalize(logMag, norm, 0, 255, NORM_MINMAX);
    norm.convertTo(norm, CV_8U);
    Mat viz;
    cvtColor(norm, viz, COLOR_GRAY2BGR);

    for (size_t i = 0; i < peaks.size(); ++i) {
        const Point& p = peaks[i].loc;
        circle(viz, p, 6, Scalar(0, 0, 255), 2);
        const std::string label = std::to_string(windows[i]);
        putText(viz, label, p + Point(5, -5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1, LINE_AA);
    }
    return viz;
}

// -----------------------------------------------------------------------------
// resolveSavePath - Maps user input to a usable file path (adds .png if needed).
// preconditions:
//   - 'requested' may be empty, a directory, or a filename.
// postconditions:
//   - Returns "" if save should be skipped; otherwise a path with extension.
static std::string resolveSavePath(const std::string& requested, const std::string& defaultName) {
    if (requested.empty()) return {};
    fs::path p(requested);
    if (fs::is_directory(p)) p /= defaultName;
    if (!p.has_extension()) p.replace_extension(".png");
    return p.string();
}

// -----------------------------------------------------------------------------
// main - Analyzes an image's FFT to suggest LP-SIFT window sizes.
// preconditions:
//   - argv[1]: image path.
//   - Optional argv[2]: save path for spectrum visualization ('' to skip).
//   - Optional argv[3]: number of peaks (default 2).
//   - Optional flag '--show' to display the spectrum window.
// postconditions:
//   - Prints peak info and suggested window sizes.
//   - Saves visualization if requested; optionally shows it.
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: fft_window_tool <image> [save_path] [num_peaks] [--show]\n"
                     "  Finds strongest frequency components and suggests window sizes.\n";
        return 1;
    }

    const std::string imgPath = argv[1];
    const std::string requestedSave = (argc >= 3) ? argv[2] : std::string();
    const std::string savePath = resolveSavePath(requestedSave, "fft_spectrum.png");
    int numPeaks = 2;
    if (argc >= 4) {
        try {
            numPeaks = std::max(1, std::stoi(argv[3]));
        } catch (...) {
            numPeaks = 2;
        }
    }
    const bool showWindow = (argc >= 5 && std::string(argv[4]) == "--show");

    Mat img = loadImageOrFail(imgPath);
    if (img.empty()) return 1;

    Mat floatImg;
    img.convertTo(floatImg, CV_32F);

    // Optionally pad to optimal DFT size
    Mat padded;
    const int m = getOptimalDFTSize(floatImg.rows);
    const int n = getOptimalDFTSize(floatImg.cols);
    copyMakeBorder(floatImg, padded, 0, m - floatImg.rows, 0, n - floatImg.cols, BORDER_CONSTANT, Scalar::all(0));

    // Create complex plane and run DFT
    Mat planes[] = {padded, Mat::zeros(padded.size(), CV_32F)};
    Mat complexImg;
    merge(planes, 2, complexImg);
    dft(complexImg, complexImg);

    fftShift(complexImg);
    split(complexImg, planes);
    magnitude(planes[0], planes[1], planes[0]);
    Mat mag = planes[0];

    mag += Scalar::all(1);
    log(mag, mag);

    // Zero out low-frequency center to ignore DC
    const Point center(mag.cols / 2, mag.rows / 2);
    circle(mag, center, 4, Scalar(0), -1);

    // Find peaks
    const int suppressRadius = 6;
    std::vector<Peak> peaks = findTopPeaks(mag.clone(), numPeaks, suppressRadius);

    std::vector<int> windows;
    windows.reserve(peaks.size());
    for (const auto& p : peaks) {
        const int dx = p.loc.x - center.x;
        const int dy = p.loc.y - center.y;
        windows.push_back(suggestWindow(dx, dy, img.rows, img.cols));
    }

    // Report results
    std::cout << "Image: " << imgPath << " (" << img.cols << "x" << img.rows << ")\n";
    for (size_t i = 0; i < peaks.size(); ++i) {
        const auto& p = peaks[i];
        const int dx = p.loc.x - center.x;
        const int dy = p.loc.y - center.y;
        const double radius = std::sqrt(static_cast<double>(dx * dx + dy * dy));
        std::cout << "Peak " << i + 1 << ": mag=" << std::fixed << std::setprecision(2) << p.mag
                  << " at (" << p.loc.x << "," << p.loc.y << "), radius=" << radius
                  << " -> window " << windows[i] << "\n";
    }
    if (peaks.empty()) {
        std::cout << "No peaks found.\n";
    }

    // Visualization
    Mat viz = makeSpectrumViz(mag, peaks, windows);
    if (!savePath.empty()) {
        try {
            imwrite(savePath, viz);
            std::cout << "Saved spectrum visualization to " << savePath << "\n";
        } catch (const cv::Exception& e) {
            std::cerr << "Failed to save spectrum visualization to " << savePath << ": " << e.what() << "\n";
        }
    }
    if (showWindow) {
        namedWindow("FFT Spectrum", WINDOW_NORMAL);
        imshow("FFT Spectrum", viz);
        waitKey(0);
    }

    return 0;
}
