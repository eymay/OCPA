#include <stdlib.h>
#include <system_error>

#include "llvm/Support/CommandLine.h"

#include "ocpa.h"
#include "util.h"

static llvm::cl::opt<std::string>
    FeaturePath("feature", llvm::cl::desc("Specify the feature file path"),
                llvm::cl::value_desc("file"), llvm::cl::Required);

static llvm::cl::opt<std::string>
    KernelPath("kernel", llvm::cl::desc("Specify the kernel file path"),
               llvm::cl::value_desc("file"), llvm::cl::Required);

static llvm::cl::opt<std::string>
    OutputPath("output", llvm::cl::desc("Specify the output file name"),
               llvm::cl::value_desc("file"), llvm::cl::Optional,
               llvm::cl::init(""));

static llvm::cl::opt<int>
    batch_size("batch_size", llvm::cl::desc("Specify the output file name"),
               llvm::cl::value_desc("Integer"), llvm::cl::Required);

int main(int argc, char **argv) {

  llvm::cl::ParseCommandLineOptions(argc, argv, "Batched ECR");

  checkPaths<std::ifstream>(FeaturePath);
  checkPaths<std::ifstream>(KernelPath);

  auto [featureWidth, featureHeight] = GetMatrixDimensions(FeaturePath);
  auto [kernelWidth, kernelHeight] = GetMatrixDimensions(KernelPath);

  OCPA_DEBUG(std::cout << "Feature matrix dimensions: " << featureWidth << " x "
                       << featureHeight << "\n";
             std::cout << "Kernel matrix dimensions: " << kernelWidth << " x "
                       << kernelHeight << "\n");

  // Data that will be passed to the GPU
  Matrix input(featureWidth, featureHeight, batch_size);
  Matrix kernel(kernelWidth, kernelHeight, batch_size);

  int stride_width = 1;

  HostData host(input, kernel, stride_width);

  Matrix tempInput(featureWidth, featureHeight);
  Matrix tempKernel(kernelWidth, kernelHeight);
  tempInput.allocateMemory();
  tempKernel.allocateMemory();

  if (!loadMatrixData(FeaturePath, tempInput) ||
      !loadMatrixData(KernelPath, tempKernel)) {
    std::cerr << "Failed to load matrix data." << std::endl;
    return 1;
  }

  // Copy the single feature and kernel into the batched matrices
  for (int i = 0; i < batch_size; ++i) {
    std::copy(tempInput.data, tempInput.data + featureWidth * featureHeight,
              host.input.data + i * featureWidth * featureHeight);
    std::copy(tempKernel.data, tempKernel.data + kernelWidth * kernelHeight,
              host.kernel.data + i * kernelWidth * kernelHeight);
  }

  if (!runECR(input, kernel, host, stride_width, batch_size)) {
    std::cerr << "Error: runSingleECR failed.\n";
    return 1;
  }

  OCPA_DEBUG(
      if (!OutputPath.empty()) {
        checkPaths<std::ofstream>(OutputPath);
        std::ofstream outputFile(OutputPath);
        writeMatrixData(host.output, outputFile);
        outputFile.close();
      } else { writeMatrixData(host.output, std::cout); })
  std::cout << "Measured time: " << host.time << "\n";

  return 0;
}
