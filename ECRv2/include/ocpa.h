#ifndef OCPA_H
#define OCPA_H

enum class calcMethod { ECR, cuDNN };
enum class cudnnAlgo { UNDEFINED, GEMM, IMPLICIT_GEMM, FFT_TILING, FAST };

struct Matrix {
  float *data;
  int width;
  int height;
  int batch_size;

  Matrix(int w, int h) : data(nullptr), width(w), height(h), batch_size(1) {}
  Matrix(int w, int h, int batch_size)
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
    this->input.data = new float[input.width * input.height * input.batch_size];
    this->kernel.data =
        new float[kernel.width * kernel.height * kernel.batch_size];
    output.data = new float[output.width * output.height * output.batch_size];
  }

  ~HostData() {
    delete[] input.data;
    delete[] kernel.data;
    delete[] output.data;
  }
};

bool runECR(Matrix &input, Matrix &kernel, HostData &host, int stride_width,
            int batch_size);

bool runCUDNN(Matrix &input, Matrix &kernel, HostData &host, int stride_width,
              int batch_size, cudnnAlgo cudnnAlgo);

#endif // OCPA_H
