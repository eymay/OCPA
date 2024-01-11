cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(OCPA LANGUAGES C CXX CUDA)

# Find the LLVM package
find_package(LLVM 16 REQUIRED CONFIG)
find_package(CUDA REQUIRED)

# Include LLVM directories
include_directories(${LLVM_INCLUDE_DIRS} "${PROJECT_SOURCE_DIR}/include")
add_definitions(${LLVM_DEFINITIONS})
include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Automatically include all source files in the src directory
file(GLOB_RECURSE SRC_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*.cu")

add_executable(singleOCPA ${SRC_FILES} tools/singleOCPA.cpp)
add_executable(batchedOCPA ${SRC_FILES} tools/batchedOCPA.cpp)

# Use CMake's built-in support to find the necessary LLVM components
# Here we look for support and core which are typically required
llvm_map_components_to_libnames(llvm_libs support)

# Link LLVM libraries to your project
# target_link_libraries(${PROJECT_NAME} ${llvm_libs})
target_link_libraries(singleOCPA ${llvm_libs} cudnn ${CUDA_LIBRARIES})
target_link_libraries(batchedOCPA ${llvm_libs} cudnn ${CUDA_LIBRARIES})

# Set the compute capability for NVIDIA GPUs
set(CUDA_ARCHITECTURES 75)

# Add the compute capability to your targets
set_property(TARGET singleOCPA PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
set_property(TARGET batchedOCPA PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
