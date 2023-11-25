
To use LLVM library together with CUDA CMake can be used:
## Requirements 
- ninja
- cmake
- clang and lld
- llvm

```shell
sudo apt install ninja-build cmake clang-16 lld-16 llvm-16
```

## Build

```shell
mkdir build && cd build

cmake -G Ninja -S .. -B . \
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_CXX_COMPILER=clang++-16 \
-DCMAKE_CUDA_COMPILER=clang++-16 \
-DCMAKE_LINKER=lld

ninja
```
## Usage
Feature and kernel paths can be provided via command line arguments. Optionally, output directory can be provided for the resulting convolution.

```shell
./singleECR --kernel <path-to-kernel>  --feature <path-to-feature>
```
For example,

```shell
./singleECR --kernel ../../dataset/resnet/kernel/layer3.2.conv2.weight  --feature ../../dataset/resnet/feature/feature38
```


