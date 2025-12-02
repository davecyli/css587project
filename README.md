# CSS587 Final Project: Local Peaks SIFT (LP-SIFT)
Authors: David Li, Ben Schipunov, Kris Yu 

# Summary
This project looks to explore and re-implement LP-SIFT in C++ from this paper: https://arxiv.org/abs/2405.08578

# OpenCV installation with xfeatures2d (needed for SURF)

Assuming folder layout:

```
C:/dev/opencv-4.x/               (OpenCV source)
C:/dev/opencv_contrib-4.x/      (opencv_contrib source)
C:/dev/opencv_build/            (build folder)
C:/dev/opencv_install/          (install folder)
```

The build folder must be separate from the source.

### Step 1 — Clone or download both repositories

Within your desired directory

```
cd C:/dev
git clone https://github.com/opencv/opencv.git opencv-4.x
git clone https://github.com/opencv/opencv_contrib.git opencv_contrib-4.x
```

### Step 2 — Create build directory

Within your desired directory

```
mkdir C:/dev/opencv_build
cd C:/dev/opencv_build
```

### Step 3 — Configure with CMake

PowerShell:

```
cmake `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DOPENCV_EXTRA_MODULES_PATH="C:/dev/opencv_contrib-4.x/modules" `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_INSTALL_PREFIX="C:/dev/opencv_install" `
  -DBUILD_opencv_world=ON `
  -DBUILD_opencv_dnn=OFF `
  -DOPENCV_ENABLE_NONFREE=ON `
  "C:/dev/opencv-4.x"
```

Bash:

```
cmake `
  -G "Visual Studio 18 2026" \
  -DOPENCV_EXTRA_MODULES_PATH="C:/dev/opencv_contrib-4.x/modules" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="C:/dev/opencv_install" \
  -DBUILD_opencv_world=ON \
  -DBUILD_opencv_dnn=OFF \
  -DOPENCV_ENABLE_NONFREE=ON \
  "C:/dev/opencv-4.x
```

Notes:

- Set -G to whichever compiler you would like to use
- DBUILD_opencv_world would be on to only include one core OpenCV lib file
- DBUILD_opencv_dnn is off to not interfere with world build
- DOPENCV_ENABLE_NONFREE is on to allow patented classes like SURF to run

### Step 4: Build Debug and Release Configs

(same location as step 3)

```
cmake --build . --config Debug --target INSTALL
```

```
cmake --build . --config Release --target INSTALL
```

### Step 5: Linking to OpenCV build

In Visual Studio Project Properties:

- **C/C++ > General > Additional Include Directories**: C:\dev\opencv_install\include

- **Linker > General > Additional Library Directories**: C:\dev\opencv_install\x64\vc18\lib (may be vc17 for VS 2022)

- **Linker > Input > Additional Dependencies**: e.g., opencv_world4130d.lib; opencv_xfeatures2d4130.lib (whichever is in the lin directory)

- **Add to System PATH**: C:\dev\opencv_install\x64\vc18\bin

# Quick start
Clone the repo
```
git clone https://github.com/davecyli/css587project.git
```
Build
```
cmake --build cmake-build-debug
```
Execute
```
Usage:
  ./css587project                           Run visual demo on all image sets
  ./css587project --benchmark               Run full benchmark, save to results.csv
  ./css587project --benchmark -o <file>     Run benchmark with custom output file
  ./css587project --benchmark --save-images Save stitched images during benchmark
  ./css587project <set1> <set2> ...         Run demo on specific image sets
  ./css587project --help                    Show this help message

Options:
  --benchmark       Run performance benchmarks instead of visual demo
  -o, --output      Specify output CSV file (default: results.csv)
  --save-images     Save stitched images during benchmark
  --no-display      Skip visual display in demo mode
```
