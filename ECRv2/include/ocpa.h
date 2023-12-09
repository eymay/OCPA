#ifndef OCPA_H
#define OCPA_H

#include "cuda_fp16.h"

enum class calcMethod { ECR, cuDNN };
enum class floatType { floatType, halfType };
enum class cudnnAlgo { UNDEFINED, GEMM, IMPLICIT_GEMM, FFT_TILING, FAST };

struct Matrix {
  float *data;
  int width;
  int height;
  int batch_size;

  Matrix(int w, int h, int batch_size = 1)
      : data(nullptr), width(w), height(h), batch_size(batch_size) {}

  void allocateMemory() { data = new float[width * height * batch_size]; }

  ~Matrix() {
    if (!data)
      delete[] data;
  }
};

struct HostData {
  Matrix input;
  Matrix kernel;
  Matrix output;
  float time;

  HostData(const Matrix &input, const Matrix &kernel, int stride_width)
      : input(input.width, input.height), kernel(kernel.width, kernel.height),
        output((input.width - kernel.width) / stride_width + 1,
               (input.height - kernel.height) / stride_width + 1),
        time(0.0) {
    this->input.allocateMemory();
    this->kernel.allocateMemory();
    output.allocateMemory();
  }

  ~HostData() {
    delete[] input.data;
    delete[] kernel.data;
    delete[] output.data;
  }
};

struct HalfMatrix {
  half *data;
  int width;
  int height;
  int batch_size;

  HalfMatrix(int w, int h, int batch_size = 1)
      : data(nullptr), width(w), height(h), batch_size(batch_size) {}

  void allocateMemory() { data = new half[width * height * batch_size]; }

  ~HalfMatrix() {
      delete[] data;
  }
};

struct HalfHostData {
  HalfMatrix input;
  HalfMatrix kernel;
  HalfMatrix output;
  float time;

  HalfHostData(const HalfMatrix &input, const HalfMatrix &kernel, int stride_width)
      : input(input.width, input.height), kernel(kernel.width, kernel.height),
        output((input.width - kernel.width) / stride_width + 1,
               (input.height - kernel.height) / stride_width + 1),
        time(0.0) {
    this->input.allocateMemory();
    this->kernel.allocateMemory();
    output.allocateMemory();
  }
  HalfHostData(const Matrix &input, const Matrix &kernel, int stride_width)
      : input(input.width, input.height), kernel(kernel.width, kernel.height),
        output((input.width - kernel.width) / stride_width + 1,
               (input.height - kernel.height) / stride_width + 1),
        time(0.0) {
    this->input.allocateMemory();
    this->kernel.allocateMemory();
    output.allocateMemory();
  }

  ~HalfHostData() {
    delete[] input.data;
    delete[] kernel.data;
    delete[] output.data;
  }
};

bool runECR(HostData &host, int stride_width,
            int batch_size);

bool runCUDNN(HostData &host, int stride_width,
              int batch_size, cudnnAlgo cudnnAlgo);
bool runCUDNN(HalfHostData &host, int stride_width,
              int batch_size, cudnnAlgo cudnnAlgo);

#endif // OCPA_H
