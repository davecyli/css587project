cmake_minimum_required(VERSION 3.16)

project(LP_SIFT_FinalProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

add_executable(lp_sift
        main.cpp
)

target_link_libraries(lp_sift PRIVATE ${OpenCV_LIBS})

# Copy images/ to the executable's output directory after build
add_custom_command(
        TARGET lp_sift POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/images"
        "$<TARGET_FILE_DIR:lp_sift>/images"
)
