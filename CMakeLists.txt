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

# Copy images into the built binary's directory (handles Debug/Release via TARGET_FILE_DIR)
add_custom_command(TARGET css587project POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:css587project>/images
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/images
            $<TARGET_FILE_DIR:css587project>/images
    COMMENT "Copying images assets next to the executable")
