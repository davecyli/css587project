cmake_minimum_required(VERSION 3.16)

project(css587project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

add_executable(css587project
    main.cpp
    lpsift.cpp
    benchmark.cpp
)

target_include_directories(css587project PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(css587project PRIVATE ${OpenCV_LIBS})

# Define the destination directory variable
set(OUTPUT_IMAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/images")

# Delete the output directory if it exists.
# REMOVE_RECURSE is safe to call even if the path doesn't exist.
file(REMOVE_RECURSE "${OUTPUT_IMAGE_DIR}")

# Copy the images folder from source to the binary output directory
# This command automatically creates the destination path if needed.
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/images" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

