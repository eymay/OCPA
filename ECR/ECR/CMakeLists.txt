cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(batchedECR LANGUAGES C CXX CUDA)

# Find the LLVM package
find_package(LLVM REQUIRED CONFIG)

# Include LLVM directories
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(${PROJECT_NAME} batchedECR.cu cl.cpp)

# Use CMake's built-in support to find the necessary LLVM components
# Here we look for support and core which are typically required
llvm_map_components_to_libnames(llvm_libs support )

# Link LLVM libraries to your project
target_link_libraries(${PROJECT_NAME} ${llvm_libs})
