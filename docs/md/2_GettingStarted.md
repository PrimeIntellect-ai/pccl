# Getting started

## Installation / Build

## Prerequisites

- Git
- CMake (3.22.1 or higher)
- C++ compiler with C++20 support (MSVC 17+, gcc 11+ or clang 12+)
- Python 3.12+ (if bindings are used)
- NVIDIA CUDA Computing Toolkit v12+ (if building with CUDA support)

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

In this section we propose a method of installing the required prerequisites for building PCCL on Windows, macOS and
Linux.

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

##### Installing CUDA on Windows

Go to https://developer.nvidia.com/cuda-downloads and download & click through the CUDA Toolkit installer.

**CAUTION:** It is crucial to do this installation process *after* installing Visual Studio because the installer will
install the "Visual Studio Integration" only if VisualStudio is installed
at the point of running the installer.
If you just freshly installed Visual Studio and still have an old CUDA Toolkit installation lying around, you will have
to uninstall and reinstall CUDA afterwards.
Make sure "Visual Studio Integration" is status "Installed" in the summary of the installer.
Without the Visual Studio Integration of the CUDA Toolkit, the cmake generation phase will fail in a specific way that
is documented in the build section below.

![Verify CUDA Installation](../images/cuda_install_visualstudio_integration_confirmation.png)

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

##### Installing CUDA (if not already installed)

The NVIDIA CUDA Computing Toolkit can be installed using any prevalent method as long as `nvcc` ends up in the system
`PATH`
of the shell that performs the cmake build.

###### Install using nvidia provided apt repository

This is the recommended way to install the CUDA toolkit on Ubuntu.
If you do not have a good reason to deviate from this (such as custom drivers, as the p2p geohot driver), you should
likely stick to this method.

Go to https://developer.nvidia.com/cuda-downloads and follow the instructions provided.

###### Install using .run file (not recommended on Ubuntu)

It is also possible to install CUDA via the NVIDIA provided .run file.
This is not the recommended way to install cuda, but there may be good reasons why not to use the system packages,
such as using a custom-built nvidia driver, such as the p2p geohot driver.
Any nvidia driver related package such as userspace libraries and the CUDA toolkit may bring kernel module dependencies
which may be undesirable. When installing via this method, carefully validate the "kernel-module-version-like
designation" (e.g. "565.57.01")
of 1.) Your cuda installation against the that of your installed driver and 2.) Your installed userspace libraries.
Your userspace libraries and kernel modules must match EXACTLY, whereas for cuda distributions the driver version must
merely exceed the minimum stated by the distribution.
Note that the .run file may replace your driver & user space library installation, if not explicitly disabled and can
conflict with installed nvidia apt packages.
Also note that seemingly innocent packages such as nvitop will bring in nvidia related apt dependencies which can result
in
broken system state and conflicts with your existing driver/userspace libraries/cuda libraries installation when using
the .run file method.
Use this method with caution and be aware that many packages are operating under the assumption that you are missing
dependencies!

```
# search for "Driver Version" to find e.g. "565.57.01"; Ensure this matches everywhere.
nvidia-smi

# check version postfix to validate userspace library installation, e.g. "libnvidia-ml.so.565.57.01" -> "565.57.01"; this must match your driver version exactly.
ls /usr/lib/x86_64-linux-gnu/ | grep libnvidia-ml.so

# download cuda .run file; ensure the driver version is higher or equal to the designation, here (driver+userspacelibs) "565.57.01" > (cuda) "560.35.05", hence ok.
wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run
sudo bash cuda_12.6.3_560.35.05_linux.run

# finally, add /usr/local/cuda/bin to $PATH with your method of choice. nvcc should be invokable from the shell used for building.
```

### Building the native library & other targets

To build all native targets, run the following commands valid for both Windows with PowerShell and Unix-like systems
starting from the root directory of the repository:

```bash
git submodule update --init --recursive
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPCCL_BUILD_CUDA_SUPPORT=ON .. # use -DPCCL_BUILD_CUDA_SUPPORT=OFF if building without cuda support
cmake --build . --config Release --parallel
```

**CAUTION:** When building on Windows, make sure to use the "x64 Native Tools Command Prompt for VS 2022". Make sure it
is specifically the 'x64' variant!

If you are getting the following error on **Windows**, the Visual Studio Integration from CUDA is missing, which is an
indication that CUDA was installed BEFORE
VisualStudio/MSVC was installed on the system:

```
-- CUDA Toolkit was found!
CMake Error at C:/Program Files/CMake/share/cmake-3.31/Modules/CMakeDetermineCompilerId.cmake:614 (message):
  No CUDA toolset found.
Call Stack (most recent call first):
  C:/Program Files/CMake/share/cmake-3.31/Modules/CMakeDetermineCompilerId.cmake:8 (CMAKE_DETERMINE_COMPILER_ID_BUILD)
  C:/Program Files/CMake/share/cmake-3.31/Modules/CMakeDetermineCompilerId.cmake:53 (__determine_compiler_id_test)
  C:/Program Files/CMake/share/cmake-3.31/Modules/CMakeDetermineCUDACompiler.cmake:131 (CMAKE_DETERMINE_COMPILER_ID)
  CMakeLists.txt:10 (enable_language)
```

In this case, the CUDA Toolkit needs to be uninstalled and reinstalled while ensuring the Visual Studio integration is
installed.
Refer to section "Installing CUDA on Windows" for more information.

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
pip install framework/ # make sure not to forget the trailing slash
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

## Basic Usage Workflow

Once you have built or installed PCCL, using it typically involves:

1. Launching a Master Node
2. Creating and connecting Peers
3. Performing Collective Operations
4. Shutting Down (both peers and master)

### Launching a Master Node

A *master node* is the orchestrator that tracks who has joined the run (peers) and what collective "topology" (e.g. ring
order) should be.
You can launch it in one of two ways:

#### Via the PCCL API:

```cpp
#include <pccl.h>
#include <pccl.h>

#include <thread>
#include <csignal>
#include <iostream>

#define PCCL_CHECK(status) { pcclResult_t status_val = status; if (status_val != pcclSuccess) { std::cerr << "Error: " << status_val << std::endl; exit(1); } }

static pcclMasterInstance_t* master_instance{};

void signal_handler(const int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cout << "Interrupting master node..." << std::endl; // is this signal async safe?
        PCCL_CHECK(pcclInterruptMaster(master_instance));
    }
}

int main() {
    ccoip_socket_address_t listen_address {};
    listen_address.inet.ipv4 = {0, 0, 0, 0};
    listen_address.port = 48148;

    // install signal handler for interrupt & termination signals
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    PCCL_CHECK(pcclCreateMaster(listen_address, &master_instance));
    PCCL_CHECK(pcclRunMaster(master_instance));

    PCCL_CHECK(pcclMasterAwaitTermination(master_instance));
    PCCL_CHECK(pcclDestroyMaster(master_instance));

    std::cout << "Master node terminated." << std::endl;
}
```

#### Via provided Executable

```bash
./ccoip_master
```

This tool handles the same calls internally as the example above.
Once the master is running, peers can connect to it by specifying the master's IP address and port.

### Creating and connecting Peers

Each worker or training process is a *peer*. In C/C++:

1. Initialize the PCCL library
2. Create a communicator using the master's address and a chosen "peer group" ID.
3. Connect to the master and wait until the master's state machine has accepted/acknowledged the peer.

```cpp
#include <pccl.h>
#include <iostream>

#define PCCL_CHECK(status) { pcclResult_t status_val = status; if (status_val != pcclSuccess) { std::cerr << "Error: " << status_val << std::endl; exit(1); } }

int main() {
    // 1) Initialize library
    PCCL_CHECK(cclInit());
    
    // 2) Prepare communicator creation parameters
    pcclComm_t *communicator{};
    constexpr pcclCommCreateParams_t params{
            .master_address = {
                    .inet = {
                            .protocol = inetIPv4,
                            // suppose that the master is at 10.1.2.92
                            .ipv4 = {10, 1, 2, 92}
                    },
                    .port = CCOIP_PROTOCOL_PORT_MASTER
            },
            .peer_group = 0
    };
    PCCL_CHECK(pcclCreateCommunicator(&params, &communicator));
    PCCL_CHECK(pcclConnect(communicator));
    
    std::cout << "Peer successfully connected!" << std::endl;
    
    // Now the communicator is ready to do All-Reduce, Shared State Sync, etc.
   
    // 3) Clean up
    // Eventually, destroy the communicator and deinitialize the library
    PCCL_CHECK(pcclDestroyCommunicator(communicator));
    return 0;
}
```

- `pcclConnect`: May stall until the master and other peers collectively *accept* you (in PCCL, the existing peers must
  effectively vote to integrate a new join).
  If the run is in the middle of certain operations (e.g. All-Reduce), it may take time before the system transitions to
  accept the newcomer.

- `peer_group`: All peers with the same group id form a logical "communicator group" for collective operations and
  shared state synchronization.
  If you want a certain subset of peers to only all-reduce and shared state synchronize among themselves, assign them a
  unique group id.

### A Typical Iteration Flow

Once a peer is connected to the master, it can:

1. Periodically Update / Optimize the Topology
2. Check the current world size
3. Run Collectives (e.g. All-Reduce)
4. Optionally Synchronize Shared State (e.g., model parameters)

In many distributed training loops, you'll repeat a sequence of:

#### (Optional) `pcclUpdateTopology(...)`

- This is where existing peers collectively decide to accept any newly connected peers or remove dropped ones.
- `Important:` If you call `pcclUpdateTopology` **too early** (e.g., at the very first iteration), you can inadvertently
  cause a deadlock because the pre-existing peers that just agreed to accept you just finished the `pcclUpdateTopology`
  step.
  Immediately calling `pcclUpdateTopology` again will not result in unanimous agreement - as the other peers will have
  moved on to subsequent operations -
  and run will stall.
  A common rule of thumb is to call `pcclUpdateTopology` only *after* at least one local iteration has passed, ensuring
  that the joined peer in the first iteration will call the same set of operations that pre-existing peers will still
  perform in the current
  iteration that they are in.

#### Obtain the world size via `pcclGetAttribute`

```cpp
int world_size{};
PCCL_CHECK(pcclGetAttribute(communicator, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size));
```

- *Pitfall*: When making branching decision based on world size, make sure it up-to-date *after* pcclUpdateTopology, as
  new peers may have been accepted into the run during
  this step, resulting in a dirty world size.
  Inconsistency here can lead to **deadlocks**, if the application
  logic branches based on the world size (e.g., if a peer thinks the world size is 1, but others already have the
  updated count of 2), resulting in different branches being taken and unanimous agreement not being possible anymore.

#### (Optional) `pcclOptimizeTopology(...)`

- If `world_size > 1`, you may choose to optimize the ring order. This triggers bandwidth tests among peers, potentially
  re-solving the traveling salesman problem for better throughput.
  Master determines what bandwidth information out of all possible point to point bandwidth measurements it does not yet
  know and requests the peers to measure the set of missing "cost edges".
  Note: if this problem is un-feasible, it will be solved heuristically first. A full solution may be attempted
  asynchronously in the background by the master node. When better solutions are found, they will be distributed to the
  peers in the next invocation of `pcclUpdateTopology`.

- `pcclOptimizeTopology` will establish p2p connections to newly designated neighboring peers if the topology changes in
  much the same way that `pcclOptimizeTopology` will, with the difference of
  not accepting brand-new peers into the run. However, the world size can still change when peers leave!
  It is recommended that the world size is re-obtained after invoking `pcclOptimizeTopology`.

#### The recommended pattern for a loop iteration is thus as follows:

```cpp
int world_size{};
PCCL_CHECK(pcclGetAttribute(communicator, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size));

for (uint64_t i = 0;;i++) {
    if (i > 0) {
        while (pcclUpdateTopology(comm) == pcclUpdateTopologyFailed) {
            std::cout << "[Peer] UpdateTopology failed => retrying...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        PCCL_CHECK(pcclGetAttribute(communicator, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size)); // get the new world size
    }
    
    if (world_size > 1) {
        // optimize the ring order for better throughput
        while (pcclOptimizeTopology(communicator) == pcclTopologyOptimizationFailed) {
            std::cout << "[Peer] OptimizeTopology failed => retrying...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        PCCL_CHECK(pcclGetAttribute(communicator, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size)); // get the new world size
    }
    
    if (world_size < 2) {
        std::this_thread::sleep_for(std::chrono::seconds(1)); // wait until we have at least 2 peers
        continue;
    }
    
    // ... perform shared state synchronization, all-reduce, etc.
}
```

This pattern follows best practices to avoid deadlocks and guard the training loop by the means that PCCL intends.

Note that having multiple call-sites to `pcclUpdateTopology` in a single iteration is not recommended, as it makes it
significantly
harder to ensure that peers are in unanimous agreement about the current state of the run.

### Shared state synchronization

If your application intents to take advantage of PCCL's shared state synchronization scheme to e.g. keep parameters in sync,  you can use the following pattern:
```cpp
float myModelWeights[4096] = {/* ... */};

pcclTensorInfo_t tinfo{
    .name                     = "model_weights",
    .data                     = myModelWeights,
    .count                    = 4096,
    .datatype                 = pcclFloat,
    .device_type              = pcclDeviceCpu,
    .allow_content_inequality = false
};
pcclSharedState_t sstate{
    .revision = (current_revision + 1), // must be exactly +1 from your previous revision
    .count    = 1,
    .infos    = &tinfo
};

pcclSharedStateSyncInfo_t sync_info{};
pcclResult_t sync_st = pcclSynchronizeSharedState(communicator, &sstate, &sync_info);
if (sync_st == pcclSuccess) {
    // If any mismatch was found, your local array is updated
    // with the "popular" version from other peers
    if (i > 0) {
        assert(sync_info.rx_bytes == 0); // assert no bit-diverge happened
    }
} else {
    // Possibly revision violation or peer dropout
}
```
Shared state synchronization unifies two main concerns:
- Checkpoint distribution on peer join
- Preventing peer divergence

The `allow_content_inequality` flag can be set to `true` if you want to allow for content inequality between peers.
This however should only be used if you are certain that the content inequality is acceptable for your use case.
Bit-parity among peers is more than feasible given that the ring reduce will guarantee that all peers receive
the same result.
If shared state is a deterministic function of the ring reduce result, you may assert that no
shared state divergence will occur, meaning that the number of received bytes is zero, as all peers will independently
advance to the same shared state. Thus, in the ideal case, shared state synchronization is relegated to a checkpoint synchronization
mechanism for joining peers while being a no-op most of the time.
As long as your optimizer is an element-wise expression - as is the case for most optimizers -
not even GPUs will behave indeterministically here, and you are free to assert bit-parity among peers.
You can be very certain that as long as you are using a well-known optimizer (such as `AdamW`) that all indeterminism
that you are observing is caused by improper synchronization of optimizer state or learning rate scheduler associated state.
Ensuring that a) your learning state scheduler behaves deterministically and b) all optimizer state is synchronized properly via shared state
should result in bit-identical independent advancement of model parameters, eliminating the need for a shared state retransmission.

### Performing Collective Operations

If you have at least 2 peers, you can do collective communications operations, such as an All-Reduce.
For example:

```cpp
float local_data[1024] = {/* ... */};
float result[1024]{};

pcclReduceDescriptor_t desc{
    .count = 1024,
    .op    = pcclSum,
    .tag   = 0,  // unique identifier for this operation
    .src_descriptor = {
        .datatype           = pcclFloat,
        .distribution_hint  = PCCL_NORMAL_DISTRIBUTION_HINT // hint distribution if know to potentially improve quantization
    },
    .quantization_options = {
        .quantized_datatype = pcclFloat,
        .algorithm          = pcclQuantNone
    }
};

pcclReduceInfo_t info{};
pcclResult_t result = pcclAllReduce(local_data, result, &desc, communicator, &info);
if (result == pcclSuccess) {
    // 'result' now has the sum across all peers
    // info.tx_bytes, info.rx_bytes contain stats
} else {
    // e.g. a peer dropped out => typically retry or handle
}
```

It is also possible to use the same buffer for both the source and destination argument, note however that PCCL will
have to internally allocate memory to fulfill this request.
You may want to take full control of memory management and prevent additional allocations by simply allocating memory
which PCCL will have to allocate anyway yourself.


## Basic Hello-World Example

The following is a simple example of a complete program that uses PCCL to perform an All-Reduce operation:

### hello_world.cpp (Peer Side)
```cpp
#include <pccl.h>
#include <iostream>
#include <thread>    // for sleep_for
#include <chrono>    // for seconds
#include <cstdlib>   // for exit
#include <cstring>   // for memset

// Helper macro for error-checking
#define PCCL_CHECK(stmt) do {                             \
    pcclResult_t _st = (stmt);                            \
    if (_st != pcclSuccess) {                             \
        std::cerr << "PCCL error: " << _st << '\n';       \
        std::exit(1);                                     \
    }                                                     \
} while(0)

// Hardcoded Master IP/Port
static constexpr uint8_t  MASTER_IP[4] = {127, 0, 0, 1};
static constexpr uint16_t MASTER_PORT  = 48148;

// We'll allow up to 5 distributed steps
static constexpr int MAX_STEPS = 5;

int main() {
    // 1) Initialize PCCL
    PCCL_CHECK(pcclInit());

    // 2) Create communicator
    pcclCommCreateParams_t params {
        .master_address = {
            .inet = {
                .protocol = inetIPv4,
                .ipv4 = { MASTER_IP[0], MASTER_IP[1], MASTER_IP[2], MASTER_IP[3] }
            },
            .port = MASTER_PORT
        },
        .peer_group = 0
    };
    pcclComm_t* comm = nullptr;
    PCCL_CHECK(pcclCreateCommunicator(&params, &comm));

    // 3) Connect to the master (blocking)
    std::cout << "[Peer] Connecting to master at "
              << int(MASTER_IP[0]) << "." << int(MASTER_IP[1]) << "."
              << int(MASTER_IP[2]) << "." << int(MASTER_IP[3])
              << ":" << MASTER_PORT << "...\n";
    PCCL_CHECK(pcclConnect(comm));
    std::cout << "[Peer] Connected!\n";

    // We'll have:
    //   - A local iteration counter "i" to skip updateTopology on i=0
    //   - A shared-state 'revision' in PCCL to keep all peers in step lock.
    int local_iter = 0; // for local logic

    // 4) Prepare some dummy data to place in shared state
    static float dummyWeights[8] = { 0.f }; // your model/optimizer state in real usage

    pcclTensorInfo_t tinfo{
        .name                     = "myWeights",
        .data                     = dummyWeights,
        .count                    = 8,
        .datatype                 = pcclFloat,
        .device_type              = pcclDeviceCpu,
        .allow_content_inequality = false
    };
    pcclSharedState_t sstate{
        .revision = shared_revision, // must match the current step
        .count    = 1,
        .infos    = &tinfo
    };

    int world_size{};
    PCCL_CHECK(pcclGetAttribute(comm, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size));

    // 5) Enter the training loop
    // We'll do up to MAX_STEPS. Each step => we do some ring operation and a shared-state sync.
    while (true) {
        // A) If we are not on the very llocal first iteration, update topology
        if (local_iter > 0) {
            while (pcclUpdateTopology(comm) == pcclUpdateTopologyFailed) {
                std::cout << "[Peer] UpdateTopology failed => retrying...\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            // get up-to-date world size
            PCCL_CHECK(pcclGetAttribute(comm, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size));
        }

        // C) If multiple peers are present => optionally optimize ring
        if (world_size > 1) {
            while (pcclOptimizeTopology(comm) == pcclOptimizeTopologyFailed) {
                std::cout << "[Peer] OptimizeTopology failed => retrying...\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            // get up-to-date world size
            PCCL_CHECK(pcclGetAttribute(comm, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size));
        } else {
            // alone => no ring-based operation => wait
            std::cout << "[Peer] alone => sleeping.\n";
            std::this_thread::sleep_for(std::chrono::seconds(1));
            // continue the loop to see if a new peer joined
            // next iteration => we can accept them
            local_iter++;
            continue;
        }

        // D) Example ring operation => a small All-Reduce
        float local_data[4];
        for (int k = 0; k < 4; k++) {
            local_data[k] = float(local_iter * 10 + (k + 1)); // something unique each iteration
        }
        float result_data[4] = {};

        pcclReduceDescriptor_t desc{
            .count = 4,
            .op    = pcclSum,
            .tag   = 0,
            .src_descriptor = {
                .datatype          = pcclFloat,
                .distribution_hint = PCCL_DISTRIBUTION_HINT_NONE
            },
            .quantization_options = {
                .quantized_datatype = pcclFloat,
                .algorithm          = pcclQuantNone
            }
        };
        pcclReduceInfo_t reduce_info{};

        bool all_reduce_fatal_failure = false;
        for (;;) {
            pcclResult_t red_st = pcclAllReduce(local_data, result_data, &desc, comm, &reduce_info);
            if (red_st == pcclSuccess) {
                std::cout << "[Peer] local_iter=" << local_iter
                          << ", All-Reduce => result = [ ";
                for (float val : result_data) std::cout << val << " ";
                std::cout << "], Tx=" << reduce_info.tx_bytes
                          << ", Rx=" << reduce_info.rx_bytes << "\n";
                break;
            } else {
                std::cout << "[Peer] All-Reduce fail: " << red_st << "; Retrying...\n";
                // the world size may have changed after a failed all reduce if a peer drops.
                PCCL_CHECK(pcclGetAttribute(comm, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size));
            
                // if every peer but us dropped, we'll need to accept new peers and wait until we have at least 2 peers again
                if (world_size < 2) {
                    all_reduce_fatal_failure = true;
                    break;
                }
            }
        }
        if (all_reduce_fatal_failure) {
            std::cout << "[Peer] All-Reduce failed fatally. We will wait until we have at least 2 peers again.\n";
            local_iter++;
            continue;
        }

        // Increment the shared state revision followed by subsequent sync.
        sstate.revision++;

        // PCCL enforces that "revision" must increment by exactly 1, for each pcclSynchronizeSharedState call.
        pcclSharedStateSyncInfo_t ssi{};
        pcclResult_t sst = pcclSynchronizeSharedState(comm, &sstate, &ssi);
        if (sst == pcclSuccess) {
            std::cout << "[Peer] shared_revision now " << shared_revision
                      << ", sync => tx=" << ssi.tx_bytes
                      << ", rx=" << ssi.rx_bytes << "\n";
        } else {
            std::cerr << "[Peer] shared-state sync fail: " << sst
                      << " at revision=" << shared_revision << "\n";
            break;
        }

        // F) Stop if we've done enough steps => i.e., if shared_revision >= MAX_STEPS
        //    Each peer that sees we reached that step will break out the same iteration.
        if (shared_revision >= MAX_STEPS) {
            std::cout << "[Peer] Reached revision " << shared_revision
                      << " => done.\n";
            break;
        }

        // G) local iteration increments for next loop:
        local_iter++;
    }

    // 6) Cleanup
    PCCL_CHECK(pcclDestroyCommunicator(comm));
    std::cout << "[Peer] Exiting.\n";
    return 0;
}

```

### How to launch

First, we need a running CCoIP master instance.
Here, we will use the provided `ccoip_master` executable:

```bash
./ccoip_master
```

Next, we run two instances of the `hello_world` program:

```bash
./hello_world &
./hello_world &
await
```
