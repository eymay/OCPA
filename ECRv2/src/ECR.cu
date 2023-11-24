#include <cuda_runtime.h>

#include "ocpa.h"
#include "ocpa_cuda.h"

__global__ void BatchedECR(int batch_size, int stride_width, Matrix input,
                           Matrix kernel, Matrix output) {
  // A one-dimensional grid processes a matrix, a block processes a row of the
  // matrix, a two-dimensional grid processes multiple matrices.
  const int blockId = blockIdx.y * gridDim.x + blockIdx.x;
  const int threadId = blockId * blockDim.x + threadIdx.x;

  // Every block has a shared memory.
  __shared__ float F_data[5120];
  __shared__ float K_data[5120];
  __shared__ int ptr[512];

  // Out-of-bounds judgement
  if (threadId < batch_size * output.height * output.width) {
    // Construct ECR storage
    int temp = 0;
    for (int i = 0; i < kernel.height; i++) {
      for (int j = 0; j < kernel.width; j++) {
        int offset = threadIdx.x + i * input.width + j;
        offset =
            offset + blockIdx.y * (kernel.height - stride_width) * input.width;
        offset = offset + stride_width * blockId * input.width +
                 threadIdx.x * (stride_width - 1);

        float value = input.data[offset];
        float kvalue = kernel.data[i * kernel.width + j];

        if ((value != 0) && (kvalue != 0)) {
          // One thread fills kernel.width * kernel.height spaces from front to
          // back
          F_data[threadIdx.x * kernel.width * kernel.height + temp] = value;
          K_data[threadIdx.x * kernel.width * kernel.height + temp] = kvalue;
          temp++;
        }
      }
    }
    if (temp != 0)
      ptr[threadIdx.x] = temp;
    else
      ptr[threadIdx.x] = -1;

    __syncthreads();

    // Convolution algorithm for ECR
    if (ptr[threadIdx.x] == -1)
      output.data[threadId] = 0;
    else
      for (int i = 0; i < ptr[threadIdx.x]; i++) {
        output.data[threadId] +=
            F_data[threadIdx.x * kernel.width * kernel.height + i] *
            K_data[threadIdx.x * kernel.width * kernel.height + i];
      }
  }
}

bool runSingleECR(Matrix &input, Matrix &kernel, HostData &host,
                  int stride_width) {
  if (!host.input.data || !host.kernel.data) {
    std::cerr << "Input or kernel data is not allocated on the host\n";
    return false;
  }
  Matrix output(host.output.width, host.output.height);

  // Allocate memory on GPU for input, kernel, and output
  checkCudaErrors(
      cudaMalloc(&input.data, input.width * input.height * sizeof(float)));
  checkCudaErrors(
      cudaMalloc(&kernel.data, kernel.width * kernel.height * sizeof(float)));
  checkCudaErrors(
      cudaMalloc(&output.data, output.width * output.height * sizeof(float)));

  // Copy input and kernel data from host to device
  // Assuming h_input and h_kernel are arrays containing the input and kernel
  // data on the host
  checkCudaErrors(cudaMemcpy(input.data, host.input.data,
                             input.width * input.height * sizeof(float),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(kernel.data, host.kernel.data,
                             kernel.width * kernel.height * sizeof(float),
                             cudaMemcpyHostToDevice));

  dim3 grid(output.height, 1/*batch size*/);
  dim3 block(output.width);

  BatchedECR<<<grid, block>>>(1 /*batch size*/, stride_width, input, kernel,
                              output);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Copy the output back to host memory
  checkCudaErrors(cudaMemcpy(host.output.data, output.data,
                             output.width * output.height * sizeof(float),
                             cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(input.data));
  checkCudaErrors(cudaFree(kernel.data));
  checkCudaErrors(cudaFree(output.data));

  return true;
}
