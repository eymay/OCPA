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

int main(int argc, char **argv) {

  llvm::cl::ParseCommandLineOptions(argc, argv, "Single ECR");

  checkPaths<std::ifstream>(FeaturePath);
  checkPaths<std::ifstream>(KernelPath);

  auto [featureWidth, featureHeight] = GetMatrixDimensions(FeaturePath);
  auto [kernelWidth, kernelHeight] = GetMatrixDimensions(KernelPath);

  OCPA_DEBUG(std::cout << "Feature matrix dimensions: " << featureWidth << " x "
                       << featureHeight << "\n";
             std::cout << "Kernel matrix dimensions: " << kernelWidth << " x "
                       << kernelHeight << "\n");

  // Data that will be passed to the GPU
  Matrix input(featureWidth, featureHeight);
  Matrix kernel(kernelWidth, kernelHeight);

  int stride_width = 1;

  HostData host(input, kernel, stride_width);

  if (!loadMatrixData(FeaturePath, host.input) ||
      !loadMatrixData(KernelPath, host.kernel)) {
    std::cerr << "Failed to load matrix data." << std::endl;
    return 1;
  }

  if (!runECR(input, kernel, host, stride_width, 1 /*batch size*/)) {
    std::cerr << "Error: runSingleECR failed.\n";
    return 1;
  }

  if (!OutputPath.empty()) {
    checkPaths<std::ofstream>(OutputPath);
    std::ofstream outputFile(OutputPath);
    writeMatrixData(host.output, outputFile);
    outputFile.close();
  } else {
    writeMatrixData(host.output, std::cout);
  }
  std::cout << "Measured time: " << host.time << "\n";

  return 0;
}
