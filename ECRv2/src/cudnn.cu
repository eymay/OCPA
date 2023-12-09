#include "cuda_fp16.h"
#include "cudnn.h"

#include "ocpa.h"
#include "ocpa_cuda.h"

bool runCUDNN(HostData &host, int stride_width, int batch_size,
              cudnnAlgo cudnnAlgo) {

  if (!host.input.data || !host.kernel.data) {
    std::cerr << "Input or kernel is not allocated on the host\n";
    return false;
  }

  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));

  Matrix input(host.input.width, host.input.height);
  Matrix kernel(host.kernel.width, host.kernel.height);
  Matrix output(host.output.width, host.output.height);
  CudaTimer timer;

  timer.startTiming();

  const int in_n = batch_size; /* number of inputs (batch size) */
  constexpr int in_c = 1;      /* number of input feature maps */
  const int in_size = input.height * input.width * in_c * in_n;

  constexpr int filt_k = 1; /* number of output feature maps */
  constexpr int filt_c = 1; /* number of input feature maps */
  const int filt_size = kernel.height * kernel.width * filt_c * filt_k;

  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, in_n, in_c,
                                        input.height, input.width));

  CUDA_CALL(cudaMalloc(&input.data, in_n * in_c * input.height * input.width *
                                        sizeof(float)));

  cudnnFilterDescriptor_t filt_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW, filt_k, filt_c,
                                        kernel.height, kernel.width));

  CUDA_CALL(cudaMalloc(&kernel.data, filt_k * filt_c * kernel.height *
                                         kernel.width * sizeof(float)));

  // convolution
  constexpr int pad_h = 0;
  constexpr int pad_w = 0;
  constexpr int str_h = 1;
  constexpr int str_w = 1;
  constexpr int dil_h = 1;
  constexpr int dil_w = 1;

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
      conv_desc, pad_h, pad_w, str_h, str_w, dil_h, dil_w, CUDNN_CONVOLUTION,
      CUDNN_DATA_FLOAT));

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

  CUDA_CALL(
      cudaMalloc(&output.data, out_n * out_c * out_h * out_w * sizeof(float)));

  // algorithm
  cudnnConvolutionFwdAlgo_t algo;
  switch (cudnnAlgo) {
  case cudnnAlgo::GEMM:
    algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    break;
  case cudnnAlgo::IMPLICIT_GEMM:
    algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    break;
  case cudnnAlgo::FFT_TILING:
    algo = CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
    break;
  case cudnnAlgo::FAST:
    std::cerr << "FAST is not supported yet\n";
    return false;
  case cudnnAlgo::UNDEFINED:
    std::cerr << "UNDEFINED is not supported yet\n";
    return false;
  }

  // CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(
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

  cudaMemcpy(input.data, host.input.data, in_size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(kernel.data, host.kernel.data, filt_size * sizeof(float),
             cudaMemcpyHostToDevice);

  CUDNN_CALL(cudnnConvolutionForward(
      cudnn, &alpha, in_desc, input.data, filt_desc, kernel.data, conv_desc,
      algo, ws_data, ws_size, &beta, out_desc, output.data));

  int result_size = out_n * out_c * out_h * out_w;
  cudaMemcpy(host.output.data, output.data, result_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  // finalizing
  CUDA_CALL(cudaFree(ws_data));
  CUDA_CALL(cudaFree(output.data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDA_CALL(cudaFree(kernel.data));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
  CUDA_CALL(cudaFree(input.data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
  CUDNN_CALL(cudnnDestroy(cudnn));

  timer.stopTiming();
  host.time = timer.getElapsedTime();

  return true;
}

bool runCUDNN(HalfHostData &host, int stride_width, int batch_size,
              cudnnAlgo cudnnAlgo) {

  if (!host.input.data || !host.kernel.data) {
    std::cerr << "Input or kernel is not allocated on the host\n";
    return false;
  }

  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));

  HalfMatrix input(host.input.width, host.input.height);
  HalfMatrix kernel(host.kernel.width, host.kernel.height);
  HalfMatrix output(host.output.width, host.output.height);
  CudaTimer timer;

  timer.startTiming();

  const int in_n = batch_size; /* number of inputs (batch size) */
  constexpr int in_c = 1;      /* number of input feature maps */
  const int in_size = input.height * input.width * in_c * in_n;

  constexpr int filt_k = 1; /* number of output feature maps */
  constexpr int filt_c = 1; /* number of input feature maps */
  const int filt_size = kernel.height * kernel.width * filt_c * filt_k;

  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, in_n, in_c,
                                        input.height, input.width));

  CUDA_CALL(cudaMalloc(&input.data, in_n * in_c * input.height * input.width *
                                        sizeof(float) / 2));

  cudnnFilterDescriptor_t filt_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW, filt_k, filt_c,
                                        kernel.height, kernel.width));

  CUDA_CALL(cudaMalloc(&kernel.data, filt_k * filt_c * kernel.height *
                                         kernel.width * sizeof(float) / 2));

  // convolution
  constexpr int pad_h = 0;
  constexpr int pad_w = 0;
  constexpr int str_h = 1;
  constexpr int str_w = 1;
  constexpr int dil_h = 1;
  constexpr int dil_w = 1;

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
      conv_desc, pad_h, pad_w, str_h, str_w, dil_h, dil_w, CUDNN_CONVOLUTION,
      CUDNN_DATA_FLOAT));

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

  CUDA_CALL(cudaMalloc(&output.data,
                       out_n * out_c * out_h * out_w * sizeof(float) / 2));

  // algorithm
  cudnnConvolutionFwdAlgo_t algo;
  switch (cudnnAlgo) {
  case cudnnAlgo::GEMM:
    algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    break;
  case cudnnAlgo::IMPLICIT_GEMM:
    algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    break;
  case cudnnAlgo::FFT_TILING:
    algo = CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
    break;
  case cudnnAlgo::FAST:
    std::cerr << "FAST is not supported yet\n";
    return false;
  case cudnnAlgo::UNDEFINED:
    std::cerr << "UNDEFINED is not supported yet\n";
    return false;
  }

  // CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(
  //     cudnn,
  //     in_desc, filt_desc, conv_desc, out_desc,
  //     CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

  // workspace
  size_t ws_size;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
      cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size));

  half *ws_data;
  CUDA_CALL(cudaMalloc(&ws_data, ws_size));

  // perform
  float alpha = 1.f;
  float beta = 0.f;

  cudaMemcpy(input.data, host.input.data, in_size * sizeof(float) / 2,
             cudaMemcpyHostToDevice);
  cudaMemcpy(kernel.data, host.kernel.data, filt_size * sizeof(float) / 2,
             cudaMemcpyHostToDevice);

  CUDNN_CALL(cudnnConvolutionForward(
      cudnn, &alpha, in_desc, input.data, filt_desc, kernel.data, conv_desc,
      algo, ws_data, ws_size, &beta, out_desc, output.data));

  int result_size = out_n * out_c * out_h * out_w;
  cudaMemcpy(host.output.data, output.data, result_size * sizeof(float) / 2,
             cudaMemcpyDeviceToHost);

  // finalizing
  CUDA_CALL(cudaFree(ws_data));
  CUDA_CALL(cudaFree(output.data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDA_CALL(cudaFree(kernel.data));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
  CUDA_CALL(cudaFree(input.data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
  CUDNN_CALL(cudnnDestroy(cudnn));

  timer.stopTiming();
  host.time = timer.getElapsedTime();

  return true;
}
