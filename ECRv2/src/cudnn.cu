#include "cuda_fp16.h"
#include "cudnn.h"

#include "ocpa.h"
#include "ocpa_cuda.h"

using namespace std;

bool runCUDNN(Matrix &input, Matrix &kernel, HostData &host, int stride_width,
              int batch_size) {

  if (!host.input.data || host.kernel.data) {
    std::cerr << "Input or kernel is not allocated on the host\n";
    return false;
  }

  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));

  Matrix output(host.output.width, host.output.height);
  CudaTimer timer;

  timer.startTiming();

  // Allocating memory for input and kernel on GPU
  checkCudaErrors(
      cudaMalloc(&input.data, input.width * input.height * sizeof(float)));
  checkCudaErrors(
      cudaMalloc(&kernel.data, kernel.width * kernel.height * sizeof(float)));
  checkCudaErrors(
      cudaMalloc(&output.data, output.width * output.height * sizeof(float)));

  const int in_n = batch_size;
  constexpr int in_c = 1;
  // in_h => height
  // in_w => width
  const int in_size = input.height * input.width * in_c * in_n;

  constexpr int filt_k = 1;
  constexpr int filt_c = 1;
  // filt_h => kernel height
  // filt_width => kernel_width
  const int file_size = kernel.height * kernel.width * filt_c * filt_k;

  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, in_n, in_c,
                                        input.height, input.width));

  float *in_data;
  CUDA_CALL(cudaMalloc(&input.data, in_n * in_c * input.height * input.width *
                                     sizeof(float)));

  cudnnFilterDescriptor_t filt_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW, filt_k, filt_c,
                                        kernel.height, kernel.width));

  float *filt_data;
  CUDA_CALL(cudaMalloc(&kernel.data, filt_k * filt_c * kernel.height *
                                       kernel.width * sizeof(float)));

  // convolution
  const int pad_h = 0;
  const int pad_w = 0;
  const int str_h = 1;
  const int str_w = 1;
  const int dil_h = 1;
  const int dil_w = 1;

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
      conv_desc, pad_h, pad_w, str_h, str_w, dil_h, dil_w, CUDNN_CONVOLUTION,
      CUDNN_DATA_FLOAT));

  // cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH);
  // output
  int out_n;
  int out_c;
  int out_h;
  int out_w;

  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
      conv_desc, in_desc, filt_desc, &out_n, &out_c, &out_h, &out_w));

  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, out_n, out_c, out_h,
                                        out_w));

  float *out_data;
  CUDA_CALL(
      cudaMalloc(&output.data, out_n * out_c * out_h * out_w * sizeof(float)));

  // algorithm
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

  // = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  // = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
  // = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
  // = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  // = CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
  // = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;

  // CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
  //     cudnn,
  //     in_desc, filt_desc, conv_desc, out_desc,
  //     CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

  // workspace
  size_t ws_size;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
      cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

  float *ws_data;
  CUDA_CALL(cudaMalloc(&ws_data, ws_size));

  // perform
  float alpha = 1.f;
  float beta = 0.f;

  cudaMemcpy(in_data, input.data, in_size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(filt_data, kernel.data, file_size * sizeof(float),
             cudaMemcpyHostToDevice);

  CUDNN_CALL(cudnnConvolutionForward(cudnn, &alpha, in_desc, in_data, filt_desc,
                                     filt_data, conv_desc, algo, ws_data,
                                     ws_size, &beta, out_desc, out_data));

  int result_size = out_n * out_c * out_h * out_w;
  float *result = new float[result_size];
  cudaMemcpy(result, out_data, result_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  // finalizing
  CUDA_CALL(cudaFree(ws_data));
  CUDA_CALL(cudaFree(out_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDA_CALL(cudaFree(filt_data));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
  CUDA_CALL(cudaFree(in_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
  CUDNN_CALL(cudnnDestroy(cudnn));

  timer.stopTiming();
  host.time = timer.getElapsedTime();

  return true;
}
