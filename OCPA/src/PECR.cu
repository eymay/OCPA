#include <cuda_runtime.h>

#include "ocpa.h"
#include "ocpa_cuda.h"

__global__ void PECR(int batch_size, int stride_width, Matrix input,
                     Matrix kernel, Matrix output) {
  // Forward inference calculation
  // Calculate a pooling result
  // int tile_size = pooling_width + (kernel_stride - 1) * (pooling_width - 1) +
  // (kernel_width - 1);

  const int blockId =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
  // const int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y *
  // blockDim.x) + threadIdx.x;
  const int threadId_block = (threadIdx.y * blockDim.x) + threadIdx.x;

  __shared__ float data[1024];
  __shared__ int index[1024];
  __shared__ int count[200];
  __shared__ float temp_pool[100];

  // Boundary check
  if (blockId < output.width * output.height * batch_size) {
    int temp = 0;
    int block_index = threadId_block * kernel.width * kernel.height;
    for (int i = 0; i < kernel.width; i++) {
      for (int j = 0; j < kernel.height; j++) {
        // unsigned int offset = (threadIdx.x % pooling.width) + (threadIdx.x /
        // pooling.width) * input.width; offset = offset + blockId
        int offset = threadIdx.x + threadIdx.y * input.width;
        offset = offset + i * input.width + j;
        offset = offset + blockIdx.y * input.width + blockIdx.x;
        offset = offset + blockIdx.z * input.width * input.height;

        if (input.data[offset] != 0) {
          int kernel_index = i * kernel.width + j;
          data[block_index + temp] = input.data[offset];
          index[block_index + temp] = kernel_index;
          temp++;
        }
      }
    }
    count[threadId_block] = temp;
    // Synchronize 4 threads of a block
    // __syncthreads();

    float temp_value = 0;
    for (int i = block_index; i < block_index + count[threadId_block]; i++) {
      temp_value += data[i] * kernel.data[index[i]];
    }
    temp_pool[threadId_block] = temp_value;
    // Synchronize the calculated values in the block
    // __syncthreads();
    __syncwarp();

    for (int reduce_stride = blockDim.x * blockDim.y / 2; reduce_stride > 0;
         reduce_stride >>= 1) {
      if (threadId_block < reduce_stride) {
        temp_pool[threadId_block] =
            (temp_pool[threadId_block] >
             temp_pool[threadId_block + reduce_stride])
                ? temp_pool[threadId_block]
                : temp_pool[threadId_block + reduce_stride];
      }
    }

    output.data[blockId] = temp_pool[0];
  }
}

bool runPECR(HostData &host, int stride_width, int batch_size) {
  if (!host.input.data || !host.kernel.data) {
    std::cerr << "Input or kernel data is not allocated on the host\n";
    return false;
  }
  // pooling
  constexpr int pooling_width = 2;
  constexpr int pooling_height = 2;
  constexpr int pooling_stride = 1;

  // hack to correct the output size for pooling
  host.output.width = (host.output.width - pooling_width) / pooling_stride + 1;
  host.output.height =
      (host.output.height - pooling_height) / pooling_stride + 1;
  host.output.batch_size = batch_size;
  delete[] host.output.data;
  host.output.allocateMemory();

  // These Matrix structs are dedicated to the GPU
  Matrix input(host.input.width, host.input.height);
  Matrix kernel(host.kernel.width, host.kernel.height);
  Matrix output(host.output.width, host.output.height);
  CudaTimer timer;
  timer.startTiming();

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

  dim3 grid(output.width, output.height, batch_size);
  dim3 block(pooling_width, pooling_height);

  PECR<<<grid, block>>>(batch_size, stride_width, input, kernel, output);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Copy the output back to host memory
  checkCudaErrors(cudaMemcpy(host.output.data, output.data,
                             output.width * output.height * sizeof(float),
                             cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaFree(input.data));
  checkCudaErrors(cudaFree(kernel.data));
  checkCudaErrors(cudaFree(output.data));

  timer.stopTiming();
  host.time = timer.getElapsedTime();

  return true;
}
