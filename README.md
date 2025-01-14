# Prime Collective Communications Library (PCCL)

The Prime Collective Communications Library (PCCL) implements efficient and fault tolerant collective communications
operations such as reductions over IP and provides shared state synchronization mechanisms to keep peers in sync and
allow for the dynamic joining and leaving of peers at any point during training.
PCCL implements a novel TCP based network protocol "Collective Communications over IP" (CCoIP). A specification for this
protocol will be released upon stabilization of the feature set.

## Prerequisites

- Git
- CMake (3.22.1 or higher)
- C++ compiler with C++20 support (MSVC 17+, gcc 11+ or clang 12+)
- Python 3.12+ (if bindings are used)

## Supported Operating Systems

- Windows
- macOS
- Linux

## Supported architectures

PCCL aims to be compatible with all architectures. While specialized kernels exist to optimize crucial operations like
CRC32 hashing and quantization, fallback to a generic implementation should always be possible.
Feel free to create issues for architecture-induced compilation failures.

### Explicitly supported are:

- x86_64
- aarch64 (incl. Apple Silicon)

## Building

### Installing prerequisites

In this section we propose a method of installing the required prerequisites for building PCCL on Windows, macOS and Linux.

#### Windows

With the winget package manager installed & up-to-date from the Microsoft Store, you can install the prerequisites as
follows:

```bash
winget install Microsoft.VisualStudio.2022.Community --silent --override "--wait --quiet --add ProductLang En-us --add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended"
winget install Git.Git --source winget
winget install Kitware.CMake --source winget
winget install Python.Python.3.12 --source winget # (if you want to use the Python bindings)
```

After installing these packages, make sure to refresh your PATH by restarting your explorer.exe in the Task Manager and
opening a new Terminal launched by said explorer.

#### macOS

```bash
xcode-select --install # if not already installed

# install Homebrew package manager if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install git # if not already installed by xcode command line tools
brew install cmake
brew install python@3.12 # (if you want to use the Python bindings)
```

We recommend using python distributions from Homebrew to avoid conflicts with the system python
and additionally because Homebrew python is built to allow attachment of debuggers, such as lldb and gdb
to debug both python and native code end to end.

#### Ubuntu

```bash
sudo apt update
sudo apt install -y build-essential
sudo apt install -y git
sudo apt install -y cmake

# (if you want to use the Python bindings)
sudo apt install -y python3.12 python3.12-venv python3-pip
```

### Building the native library & other targets

To build all native targets, run the following commands valid for both Windows with PowerShell and Unix-like systems
starting from the root directory of the repository:

```bash
git submodule update --init --recursive
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release --parallel
```

### Recommended way to use the pccl native library

The recommended way to use PCCL in a C/C++ project is to clone the PCCL repository and link against the pccl library in
CMake:

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

### Installing the Python Package locally

To install the Python package locally, starting from the root directory of the repository, run the following commands:

```bash
git submodule update --init --recursive # if not done already
```

#### Optionally create a virtual environment.

NOTE: Sometimes this may even be REQUIRED on certain distributions when using system python

```bash
cd python
python -m venv venv # on Linux, "python3"/"python3.12" may be required instead of "python"
```

To activate the virtual environment, depending on your operating system and shell, run one of the following commands:

```ps1
# on Windows with PowerShell
.\venv\Scripts\Activate.ps1
```

NOTE: `Set-ExecutionPolicy -ExecutionPolicy AllSigned -Scope CurrentUser` and choosing `A` for "Always Run" may be
required to run the script.

```batch
# on Windows with cmd
.\venv\Scripts\activate.bat
```

```bash
# on Unix-like systems
source venv/bin/activate
```

Then build and install the package from source:

```
pip install python/framework/ # make sure not to forget the trailing slash
```

To test the installation, run the following command valid for both Windows with PowerShell and Unix-like systems:

```bash
python -c 'import pccl; print(pccl.__version__)'
```

## Testing

### C++ Tests

To run the C++ unit tests, starting from the root directory of the repository, run the following commands valid for both
Windows with PowerShell and Unix-like systems:

```bash
cd build
ctest --verbose --build-config Release --output-on-failure
```

### Python Tests

To run the python unit and end-to-end tests, starting from the root directory of the repository, run the following
commands:

```bash
cd python/tests

# Run unit tests
cd ../unit_tests
pip install -r requirements.txt # install requirements for unit tests
python -m pytest -s

# Run end to end tests
cd end_to_end
pip install -r requirements.txt # install requirements for e2e tests
python -m pytest -s

# If you want to run pytorch only tests / numpy only tests, make sure
# to uninstall numpy or pytorch respectively before running the tests
cd ../pytorch_only_tests
pip install -r requirements.txt # install requirements for pytorch only tests
pip uninstall -y numpy
python -m pytest -s

cd ../numpy_only_tests
pip install -r requirements.txt # install requirements for numpy only tests
pip uninstall -y torch
python -m pytest -s

# Recommended: re-install both numpy and pytorch after running the tests
pip install numpy torch
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
