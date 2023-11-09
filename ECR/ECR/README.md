
To use LLVM library together with CUDA CMake can be used:
## Requirements 
- ninja
- cmake
- clang and lld
- llvm

## Usage

```shell
mkdir build

cmake -G Ninja -S .. -B . \
-DCMAKE_CXX_COMPILER=clang++-16 \
-DCMAKE_CUDA_COMPILER=clang++-16 \
-DCMAKE_LINKER=lld

ninja
```
