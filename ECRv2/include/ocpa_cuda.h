#ifndef OCPA_CUDA_H
#define OCPA_CUDA_H

#include <iostream>

// Define a macro to wrap CUDA API calls
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

// Function to check and report CUDA errors
void check_cuda(cudaError_t result, const char *const func,
                const char *const file, int const line) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA error at " << file << ":" << line
              << " code=" << static_cast<unsigned int>(result) << " \"" << func
              << "\" \n";

    exit(99);
  }
}

#endif // OCPA_CUDA_H
