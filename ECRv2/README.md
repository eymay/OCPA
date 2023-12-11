
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

For development:
```shell
mkdir build && cd build

cmake -G Ninja -S .. -B . \
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_CXX_COMPILER=clang++-16 \
-DCMAKE_CUDA_COMPILER=clang++-16 \
-DCMAKE_LINKER=lld

ninja
```

For time measurements, invoke CMake with `Release` mode:
```shell
cmake -G Ninja -S .. -B . \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_CXX_COMPILER=clang++-16 \
-DCMAKE_CUDA_COMPILER=clang++-16 \
-DCMAKE_LINKER=lld
```
# singleECR 
singleECR driver allows us to run single convolution operations per feature and kernel. When the code is built its binary is under build directory. Time measurement is printed to console.

## Usage

Feature and kernel paths can be provided via command line arguments. Optionally, output directory can be provided for the resulting convolution.

```shell
./singleECR --kernel <path-to-kernel>  --feature <path-to-feature>
```

For example,

```shell
./singleECR --kernel ../../dataset/resnet/kernel/layer3.2.conv2.weight  --feature ../../dataset/resnet/feature/feature38 --output singleECR_result.txt
```

## Testing

```shell
python3 conv_test.py --kernel ../../dataset/resnet/kernel/layer3.2.conv2.weight  --feature ../../dataset/resnet/feature/feature38 --test_output singleECR_result.txt
```

### All possible time measurement generations
```shell
./time_recorder.sh resnet ecr
```
```shell
./time_recorder.sh resnet pecr <batch-size>
```
```shell
./time_recorder.sh resnet cudnn gemm
```
```shell
./time_recorder.sh resnet cudnn imp_gemm
```
```shell
./time_recorder.sh resnet cudnn fft
```

By changing **resnet** to **vggdata** similar measurements can be obtained for vgg, **batch-size** should be set to actual batch size such as 32

After executing and recording particular time execute the below shell script from command line to clean up previously recorded time measurement files, and you can execute a new time recording command again

```shell
./measurement_cleaner.sh
```