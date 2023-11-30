#ifndef OCPA_H
#define OCPA_H

struct Matrix {
  float *data;
  int width;
  int height;

  Matrix(int w, int h) : data(nullptr), width(w), height(h) {}
  Matrix(float *data, int w, int h) : data(data), width(w), height(h) {}
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
    this->input.data = new float[input.width * input.height];
    this->kernel.data = new float[kernel.width * kernel.height];
    output.data = new float[output.width * output.height];
  }

  ~HostData() {
    delete[] input.data;
    delete[] kernel.data;
    delete[] output.data;
  }
};

bool runSingleECR(Matrix &input, Matrix &kernel, HostData &host,
                  int stride_width);

#endif // OCPA_H
