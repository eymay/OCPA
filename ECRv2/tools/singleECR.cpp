#include <stdlib.h>
#include <system_error>

#include "llvm/Support/CommandLine.h"

#include "ocpa.h"
#include "util.h"

static llvm::cl::opt<std::string>
    FeatureNamePath("feature",
                    llvm::cl::desc("Specify the feature name file path"),
                    llvm::cl::value_desc("filename"), llvm::cl::Required);

static llvm::cl::opt<std::string>
    KernelNamePath("kernel",
                   llvm::cl::desc("Specify the kernel name file path"),
                   llvm::cl::value_desc("filename"), llvm::cl::Required);

static llvm::cl::opt<std::string>
    OutputFileDir("output", llvm::cl::desc("Specify the output directory path"),
                  llvm::cl::value_desc("directory"), llvm::cl::Optional,
                  llvm::cl::init("."));

int main(int argc, char **argv) {

  llvm::cl::ParseCommandLineOptions(argc, argv, "Batched ECR Program");

  checkPaths(FeatureNamePath);
  checkPaths(KernelNamePath);

  auto [featureWidth, featureHeight] = GetMatrixDimensions(FeatureNamePath);
  auto [kernelWidth, kernelHeight] = GetMatrixDimensions(KernelNamePath);

  OCPA_DEBUG(std::cout << "Feature matrix dimensions: " << featureWidth << " x "
                       << featureHeight << "\n";
             std::cout << "Kernel matrix dimensions: " << kernelWidth << " x "
                       << kernelHeight << "\n");

  // Data that will be passed to the GPU
  Matrix input(featureWidth, featureHeight);
  Matrix kernel(kernelWidth, kernelHeight);

  int stride_width = 1;

  HostData host(input, kernel, stride_width);

  if (!loadMatrixData(FeatureNamePath, host.input) ||
      !loadMatrixData(KernelNamePath, host.kernel)) {
    std::cerr << "Failed to load matrix data." << std::endl;
    return 1;
  }

  if (!runSingleECR(input, kernel, host, stride_width)) {
    std::cerr << "Error: runSingleECR failed.\n";
    return 1;
  }

  printMatrixData(host.output);

  return 0;
}
