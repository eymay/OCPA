#ifndef CUDNN_UTIL_H
#define CUDNN_UTIL_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <cuda.h>
#include <cudnn.h>

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

#define CUDA_CALL(f)                                           \
    {                                                          \
        cudaError_t err = (f);                                 \
        if (err != cudaSuccess)                                \
        {                                                      \
            std::cout                                          \
                << "    Error occurred: " << err << std::endl; \
            std::exit(1);                                      \
        }                                                      \
    }

#define CUDNN_CALL(f)                                          \
    {                                                          \
        cudnnStatus_t err = (f);                               \
        if (err != CUDNN_STATUS_SUCCESS)                       \
        {                                                      \
            std::cout                                          \
                << "    Error occurred: " << err << std::endl; \
            std::exit(1);                                      \
        }                                                      \
    }

struct Matrix
{
    float *data;
    int feature_width;
    int feature_height;
    int batch_size;

    Matrix(int w, int h) : data(nullptr), feature_width(w), feature_height(h), batch_size(1) {}
    Matrix(int w, int h, int batch_size)
        : data(nullptr), feature_width(w), feature_height(h), batch_size(batch_size) {}

    void allocateMemory() { data = new float[feature_width * feature_height * batch_size]; }

    ~Matrix()
    {
        if (!data)
            delete[] data;
    }
};

struct HostData
{
    Matrix input;
    Matrix kernel;
    Matrix output;
    float time;

    HostData(const Matrix &input, const Matrix &kernel, int stride_width)
        : input(input.feature_width, input.feature_height), kernel(kernel.feature_width, kernel.feature_height),
          output((input.feature_width - kernel.feature_width) / stride_width + 1,
                 (input.feature_height - kernel.feature_height) / stride_width + 1),
          time(0.0)
    {
        this->input.data = new float[input.feature_width * input.feature_height * input.batch_size];
        this->kernel.data =
            new float[kernel.feature_width * kernel.feature_height * kernel.batch_size];
        output.data = new float[output.feature_width * output.feature_height * output.batch_size];
    }

    ~HostData()
    {
        delete[] input.data;
        delete[] kernel.data;
        delete[] output.data;
    }
};

// class that keeps track of passed time
// after the input and kernel matrices are
// created up until to compuations finalized on GPU
class CudaTimer
{
    cudaEvent_t start, stop;
    float elapsedTime;

public:
    CudaTimer() : elapsedTime(0.0)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~CudaTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void startTiming() { cudaEventRecord(start, 0); }

    void stopTiming()
    {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
    }

    float getElapsedTime() const { return elapsedTime; }
};

#endif