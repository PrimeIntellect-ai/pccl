# Prime Collective Communications Library (PCCL)

The Prime Collective Communications Library (PCCL) implements efficient and fault tolerant collective communications operations such as reductions over IP and provides shared state synchronization mechanisms to keep peers in sync and allow for the dynamic joining and leaving of peers at any point during training.
PCCL implements a novel TCP based network protocol "Collective Communications over IP" (CCoIP). A specification for this protocol will be released upon stabilization of the feature set.

## Prerequisites

- CMake (3.22.1 or higher)
- C++ compiler with C++20 support (MSVC 17+, gcc 11+ or clang 12+)
- Python 3.12+ (if bindings are used)

## Supported Operating Systems
- Windows
- macOS
- Linux

## Supported architectures
PCCL aims to be compatible with all architectures. While specialized kernels exist to optimize crucial operations like CRC32 hashing and quantization, fallback to a generic implementation should always be possible.
Feel free to create issues for architecture-induced compilation failures.

### Explicitly supported are:
- x86_64
- aarch64 (incl. Apple Silicon)

## Building

### Building the native library & other targets

```bash
git submodule update --init --recursive
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release --parallel
```

### Installing the Python Package locally

```bash
git submodule update --init --recursive
pip install ./python/framework
```

### Recommended way to use the pccl native library

The recommended way to use PCCL in a C/C++ project is to clone the PCCL repository and link against the pccl library in CMake:

```bash
git clone --recurse https://github.com/PrimeIntellect-ai/pccl.git
```

Then add the newly cloned repository as a subdirectory in your CMakeLists file:
```cmake
add_subdirectory(pccl)
```

Then link against the pccl library
```cmake
target_link_libraries(YourTarget PRIVATE pccl)
```

## Testing

### C++ Tests
```bash
cd build
ctest
```

### Python Tests
```bash
cd python/tests
pip install -r requirements.txt
python -m pytest
```

## Examples

The library includes several examples:
- Basic reduction operations
- DDP MNIST training example using Pytorch
- Network peer acceptance tests

For detailed examples, see:
- C++: `tests/basic_reduce_test/`
- Python: `python/tests/end_to_end/`

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please submit issues and pull requests to help improve PCCL.
