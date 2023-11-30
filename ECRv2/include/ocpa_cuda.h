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

class CudaTimer {
  cudaEvent_t start, stop;
  float elapsedTime;

public:
  CudaTimer() : elapsedTime(0.0) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~CudaTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void startTiming() { cudaEventRecord(start, 0); }

  void stopTiming() {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
  }

  float getElapsedTime() const { return elapsedTime; }
};

#endif // OCPA_CUDA_H
