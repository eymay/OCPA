#include <stdlib.h>
#include "llvm/Support/CommandLine.h"


// Define command line options using LLVM's command line parser
static llvm::cl::opt<std::string> KernelNamePath(
    "kernel-name-path",
    llvm::cl::desc("Specify the kernel name file path"),
    llvm::cl::value_desc("filename"),
    llvm::cl::Required);

static llvm::cl::opt<std::string> TimeFilePath(
    "time-file-path",
    llvm::cl::desc("Specify the time file path"),
    llvm::cl::value_desc("filename"),
    llvm::cl::Required);

static llvm::cl::opt<int> BatchSize(
    "batch-size",
    llvm::cl::desc("Specify the batch size"),
    llvm::cl::value_desc("number"),
    llvm::cl::Required);

// Declare the function defined in the CUDA file
extern void runBatchedECR(const int batch_size, const std::string& featurePath, const std::string& kernelPath);

int main(int argc, char** argv) {
    // Parse the command line arguments
    llvm::cl::ParseCommandLineOptions(argc, argv, "Batched ECR Program");

    // Call the function with the parsed arguments
    runBatchedECR(BatchSize, KernelNamePath, TimeFilePath);

    return 0;
}
