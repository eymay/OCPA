#include <stdlib.h>
#include <system_error>

#include "llvm/Support/CommandLine.h"

#include "ocpa.h"
#include "util.h"

llvm::cl::opt<calcMethod> CalcMethod(
    llvm::cl::desc("Select the calculation method:"),
    llvm::cl::values(clEnumValN(calcMethod::ECR, "ecr", "ECR calculation"),
                     clEnumValN(calcMethod::cuDNN, "cudnn",
                                "cuDNN based calculation")),
    llvm::cl::Required);

llvm::cl::opt<floatType> FloatPrecision(
    llvm::cl::desc("Select the float precision for cuDNN:"),
    llvm::cl::values(clEnumValN(floatType::floatType, "float", "Float type"),
                     clEnumValN(floatType::halfType, "half", "Half type")),
    llvm::cl::Optional, llvm::cl::init(floatType::floatType));

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

int main(int argc, char **argv) {

  llvm::cl::ParseCommandLineOptions(argc, argv, "Single ECR");

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

  Matrix input(featureWidth, featureHeight);
  Matrix kernel(kernelWidth, kernelHeight);

  constexpr int stride_width = 1;


  switch (FloatPrecision) {
  case floatType::floatType: {
    HostData host(input, kernel, stride_width);

    if (!loadMatrixData(FeaturePath, host.input) ||
        !loadMatrixData(KernelPath, host.kernel)) {
      std::cerr << "Failed to load matrix data." << std::endl;
      return 1;
    }

    switch (CalcMethod) {
    case calcMethod::ECR:
      if (!runECR(host, stride_width, 1 /*batch size*/)) {
        std::cerr << "Error: runECR failed.\n";
        return 1;
      }
      break;
    case calcMethod::cuDNN:
      if (!runCUDNN(host, stride_width, 1 /*batch size*/, cuDNNAlgo)) {
        std::cerr << "Error: run CUDNN failed.\n";
        return 1;
      }
      break;
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
    break;
}
  case floatType::halfType:
    HalfHostData halfhost(input, kernel, stride_width);
    if (!loadMatrixData(FeaturePath, halfhost.input) ||
        !loadMatrixData(KernelPath, halfhost.kernel)) {
      std::cerr << "Failed to load matrix data." << std::endl;
      return 1;
    }

    switch (CalcMethod) {
    case calcMethod::ECR:
      std::cerr << "ECR does not support half\n";
      return 1;
    case calcMethod::cuDNN:
      if (!runCUDNN(halfhost, stride_width, 1 /*batch size*/, cuDNNAlgo)) {
        std::cerr << "Error: run CUDNN failed.\n";
        return 1;
      }
      break;
    }

    if (!OutputPath.empty()) {
      checkPaths<std::ofstream>(OutputPath);
      std::ofstream outputFile(OutputPath);
      writeMatrixData(halfhost.output, outputFile);
      outputFile.close();
    } else {
      writeMatrixData(halfhost.output, std::cout);
    }
    std::cout << "Measured time: " << halfhost.time << "\n";
    break;
  }

  return 0;
}
