# CSS587 Final Project: Local Peaks SIFT (LP-SIFT)
Authors: David Li, Ben Schipunov, Kris Yu 

# Summary
This project looks to explore and re-implement LP-SIFT in C++ from this paper: https://arxiv.org/abs/2405.08578

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
