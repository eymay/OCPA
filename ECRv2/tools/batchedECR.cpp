#include <stdlib.h>
#include <system_error>

#include "llvm/Support/CommandLine.h"

#include "ocpa.h"
#include "util.h"

llvm::cl::opt<calcMethod> CalcMethod(
    llvm::cl::desc("Select the calculation method:"),
    llvm::cl::values(
        clEnumValN(calcMethod::ECR, "ecr", "ECR calculation for convolution"),
        clEnumValN(calcMethod::PECR, "pecr",
                   "PECR calculation for combined convolution and pooling"),
        clEnumValN(calcMethod::cuDNN, "cudnn", "cuDNN based calculation")),
    llvm::cl::Required);

// llvm::cl::opt<floatType> FloatPrecision(
//     llvm::cl::desc("Select the float precision for cuDNN:"),
//     llvm::cl::values(clEnumValN(floatType::floatType, "float", "Float type"),
//                      clEnumValN(floatType::halfType, "half", "Half type")),
//     llvm::cl::Optional, llvm::cl::init(floatType::floatType));

llvm::cl::opt<cudnnAlgo> cuDNNAlgo(
    llvm::cl::desc("Select the cuDNN Algorithm:"),
    llvm::cl::values(
        clEnumValN(cudnnAlgo::GEMM, "gemm", "GEMM Algorithm"),
        clEnumValN(cudnnAlgo::IMPLICIT_GEMM, "imp_gemm",
                   "Implicit GEMM Algorithm"),
        clEnumValN(cudnnAlgo::FFT_TILING, "fft", "FFT Tiling Algorithm"),
        clEnumValN(cudnnAlgo::FAST, "fast", "Fastest Algorithm is found")),
    llvm::cl::Optional, llvm::cl::init(cudnnAlgo::UNDEFINED));

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

  if (CalcMethod == calcMethod::cuDNN && cuDNNAlgo == cudnnAlgo::UNDEFINED) {
    std::cerr << "cuDNN Algorithm is not specified.\n";
    return 1;
  }

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

  constexpr int stride_width = 1;

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

  switch (CalcMethod) {
  case calcMethod::ECR:
    if (!runECR(host, stride_width, batch_size)) {
      std::cerr << "Error: runECR failed.\n";
      return 1;
    }
    break;
  case calcMethod::PECR:
    if (!runPECR(host, stride_width, batch_size)) {
      std::cerr << "Error: runECR failed.\n";
      return 1;
    }
    break;
  case calcMethod::cuDNN:
    if (!runCUDNN(host, stride_width, batch_size, cuDNNAlgo)) {
      std::cerr << "Error: run CUDNN failed.\n";
      return 1;
    }
    break;
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
