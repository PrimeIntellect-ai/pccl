# Examples & Tutorials

This section provides concrete usage examples for various scenarios—CPU-only collectives, shared-state synchronization, PyTorch integration, and fault-tolerant runs where peers may crash or drop out. For more extensive examples, see the `tests/` and `python/examples/` or `python/tests` directories in the repository.

## CPU-Only “Hello World” All-Reduce

The following is a simple example of a complete program that uses PCCL to perform an All-Reduce operation:

```cpp
#include <pccl.h>
#include <iostream>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <cstring>

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
        .revision = 0,
        .count    = 1,
        .infos    = &tinfo
    };

    // 5) Enter the training loop
    // We'll do up to MAX_STEPS. Each step => we do some ring operation and a shared-state sync.
    while (true) {
        // A) If we are not on the very llocal first iteration, update topology
        if (local_iter > 0) {
            while (pcclUpdateTopology(comm) == pcclUpdateTopologyFailed) {
                std::cout << "[Peer] UpdateTopology failed => retrying...\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        // B) get the updated world size (always after updateTopology)
        int world_size{};
        PCCL_CHECK(pcclGetAttribute(comm, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size));

        // C) If multiple peers are present => optionally optimize ring
        if (world_size > 1) {
            while (pcclOptimizeTopology(comm) == pcclOptimizeTopologyFailed) {
                std::cout << "[Peer] OptimizeTopology failed => retrying...\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            // the world size may have changed after pcclOptimizeTopology, if a peer drops.
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

        // D) Synchronize shared state.
        //  The shared state revision of new commers will be set to the popular shared state revision along with contents. 
        // PCCL enforces that "revision" must increment by exactly 1, for each pcclSynchronizeSharedState call.
        pcclSharedStateSyncInfo_t ssi{};
        pcclResult_t sst = pcclSynchronizeSharedState(comm, &sstate, &ssi);
        if (sst == pcclSuccess) {
            std::cout << "[Peer] shared state revision now " << sstate.revision
                      << ", sync => tx=" << ssi.tx_bytes
                      << ", rx=" << ssi.rx_bytes << "\n";
            // we can assert that we do not receive data beyond the initial transfer on join, where we have to obtain the popular state.
            if (local_iter > 1) {
                assert(ssi.rx_bytes == 0);
            }
        } else {
            std::cerr << "[Peer] shared-state sync fail: " << sst
                      << " at revision=" << sstate.revision << "\n";
            break;
        }

        // E) Example ring operation => a small All-Reduce
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

        // F) Stop if we've done enough steps => i.e., if revision >= MAX_STEPS
        //    Each peer that sees we reached that step will break out the same iteration.
        if (sstate.revision >= MAX_STEPS) {
            std::cout << "[Peer] Reached revision " << sstate.revision
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

1. Run a master (via `./ccoip_master`) or your own master code.
2. Launch two (or more) `hello_world` executables. Each peer will connect to the master, update the topology, and do the All-Reduce.

### Python bindings & PyTorch Integration

PCCL has python bindings an can interact with both Numpy arrays and PyTorch tensors.
The following is a simple example of a complete program that uses PCCL to perform an All-Reduce operation with PyTorch tensors:

```python
import time
import pccl
import numpy as np

# Hardcoded Master IP/Port
MASTER_HOST = "127.0.0.1"
MASTER_PORT = 48148
MAX_STEPS   = 5

def main():
    # 1) Create a Communicator object
    #    The second argument is 'peer_group' = 0
    comm = pccl.Communicator(f"{MASTER_HOST}:{MASTER_PORT}", 0)

    # 2) Connect to the master (blocking with limited attempts)
    print(f"[Peer] Connecting to master at {MASTER_HOST}:{MASTER_PORT} ...")
    comm.connect(n_attempts=15)
    print("[Peer] Connected!")

    # We'll keep:
    #   - a local iteration counter `local_iter` to skip update_topology on i=0
    #   - a shared_state 'revision' to keep all peers in step lock
    local_iter = 0

    # 3) Prepare some dummy data to place in "shared state"
    #    Suppose in real usage these are model/optimizer states
    dummy_weights = np.zeros(8, dtype=np.float32)

    # We'll store this dummy_weights in a single TensorInfo
    # The pccl.SharedState object can have multiple TensorInfo if you want to sync more keys
    my_weights_info = pccl.TensorInfo.from_numpy(dummy_weights, "dummy_weights", allow_content_inequality=False)
    
    # We'll wrap that in a SharedState
    shared_state = pccl.SharedState([my_weights_info])
    shared_state.revision = 0

    # 4) Enter the training loop
    # We'll do up to MAX_STEPS. Each step => ring operation + shared-state sync
    while True:
        # A) If we are not on the very local first iteration, call update_topology
        if local_iter > 0:
            # keep retrying if it fails
            while True:
                try:
                    comm.update_topology()
                    break
                except pccl.PCCLError as e:
                    # could be pccl.UpdateTopologyFailed or other
                    print(f"[Peer] UpdateTopology failed => {e}. Retrying...")
                    time.sleep(0.1)

            # B) get the updated world size
            #    after update_topology, it’s guaranteed to be fresh
            world_size = comm.get_attribute(pccl.Attribute.GLOBAL_WORLD_SIZE)

        # C) If multiple peers => optionally optimize ring
        if world_size > 1:
            while True:
                try:
                    comm.optimize_topology()  # may raise an error if it fails
                    break
                except pccl.PCCLError as e:
                    print(f"[Peer] OptimizeTopology failed => {e}. Retrying...")
                    time.sleep(0.1)
            # D) get the updated world size
            #    after optimize_topology, it’s guaranteed to be fresh
            world_size = comm.get_attribute(pccl.Attribute.GLOBAL_WORLD_SIZE)
        
        
        if world_size < 2:
            # alone => no ring-based operation => wait
            print("[Peer] alone => sleeping.")
            time.sleep(1)
            local_iter += 1
            continue
            
        # E) Perform shared state synchronization
        try:
            sync_info = comm.sync_shared_state(shared_state)
            print(f"[Peer] shared_revision now {shared_state.revision}, sync => "
                  f"tx={sync_info.tx_bytes}, rx={sync_info.rx_bytes}")
            if local_iter > 1:
                # be pedantic about shared state drifts
                assert sync_info.rx_bytes == 0, "Peer shared state drifted!"
        except pccl.PCCLError as e:
            print(f"[Peer] shared-state sync fail => {e} at revision={shared_state.revision}")
            # break out => no sense continuing
            break

        # F) Example ring operation => a small All-Reduce
        local_data = np.array([local_iter * 10 + (k+1) for k in range(4)], dtype=np.float32)
        result_data = np.zeros_like(local_data)

        # We define how we want to do the reduce
        # distribution_hint can be normal if you want to help quantization
        # quantization_options can be left as defaults if no compression is needed
        # We'll do a loop to keep retrying if a rank drops
        while True:
            try:
                info = comm.all_reduce(local_data, result_data,
                                       operand_descriptor=pccl.ReduceOperandDescriptor(
                                           datatype=pccl.DataType.FLOAT,
                                           distribution_hint=pccl.DistributionHint.NONE
                                       ),
                                       quantization_options=pccl.QuantizationOptions(
                                           quantized_datatype=pccl.DataType.FLOAT,
                                           algorithm=pccl.QuantizationAlgorithm.NONE
                                       ),
                                       op=pccl.ReduceOp.SUM)
                # success => break out
                print(f"[Peer] local_iter={local_iter}, All-Reduce => result={result_data.tolist()}, "
                      f"Tx={info.tx_bytes}, Rx={info.rx_bytes}")
                break
            except pccl.PCCLError as e:
                # Could be rank disconnect or partial ring failure
                print(f"[Peer] All-Reduce fail => {e}; Retrying...")
                # check if we are now alone
                world_size = comm.get_attribute(pccl.Attribute.GLOBAL_WORLD_SIZE)
                if world_size < 2:
                    print("[Peer] All-Reduce failed and now world_size < 2 => waiting until a new peer joins")
                    # We'll just break from the reduce attempt, do next iteration
                    # That can eventually call update_topology again to let new peers in
                    break
                time.sleep(0.2)

        # If we ended up alone mid-collective, skip the rest of this iteration
        world_size = comm.get_attribute(pccl.Attribute.GLOBAL_WORLD_SIZE)
        if world_size < 2:
            local_iter += 1
            continue

        # Increment the shared revision => sync
        shared_state.revision += 1

        # Stop if we've done enough steps
        if shared_state.revision >= MAX_STEPS:
            print(f"[Peer] Reached revision {shared_state.revision} => done.\n")
            break

        local_iter += 1

if __name__ == "__main__":
    main()
```

### Basic MNIST Example
```python
import os
import time
from typing import List

import pccl
from pccl import Communicator, SharedState, TensorInfo, ReduceOp, PCCLError, ReduceOperandDescriptor, \
    DataType, DistributionHint, QuantizationOptions
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set device
device = torch.device("cpu")

# Define hyperparameters
input_size = 28 * 28  # MNIST images are 28x28
# hidden_sizes = [1024, 4096, 4096, 4096, 4096, 1024]
hidden_sizes = [256]
num_classes = 10  # Digits 0-9
batch_size = 128
learning_rate = 0.001
IS_CI = os.getenv('IS_CI', '0') == '1'
max_steps = IS_CI and 512 or 2048

# MNIST dataset (images and labels)
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

# Data loader (input pipeline)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define a simple neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.relu = nn.ReLU()
        self.fcs = nn.ModuleList(
            [nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(len(hidden_sizes) - 1)])
        self.fc2 = nn.Linear(hidden_sizes[-1], num_classes)

    def forward(self, x):
        x = x.view(-1, input_size)  # Flatten the image
        x = self.fc1(x)
        x = self.relu(x)
        for fc in self.fcs:
            x = fc(x)
            x = self.relu(x)
        x = self.fc2(x)
        return x


MASTER_HOST = "127.0.0.1"
MASTER_PORT = 48148

def main():
    model = NeuralNet(input_size, hidden_sizes, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # communicator
    communicator: Communicator = Communicator(f"{MASTER_HOST}:{MASTER_PORT}", 0)
    communicator.connect(n_attempts=15)
    print(f"Connected to the master node; PID={os.getpid()}")

    # perform a dummy forward pass to initialize the optimizer state
    for p in model.parameters():
        p.grad = torch.zeros_like(p)  # set all gradients to zero
    optimizer.step()
    print(f"Initialized optimizer state")

    # Reference model and optimizer state from shared state struct
    shared_state_dict = {}
    for name, param in model.named_parameters():
        shared_state_dict[name] = param

        # Access optimizer state
        state = optimizer.state[param]
        exp_avg = state.get('exp_avg')
        exp_avg_sq = state.get('exp_avg_sq')
        step_tensor = state.get('step')

        if exp_avg is None or exp_avg_sq is None or step_tensor is None:
            raise ValueError(f"Optimizer state for parameter '{name}' is not initialized.")

        # Add optimizer state tensors with associated names
        shared_state_dict[f"{name}_m1"] = exp_avg
        shared_state_dict[f"{name}_m2"] = exp_avg_sq
        shared_state_dict[f"{name}_step"] = step_tensor

    entries = [TensorInfo.from_torch(tensor, name, allow_content_inequality=False) for name, tensor in
               shared_state_dict.items()]
    shared_state: SharedState = SharedState(entries)
    print(f"Initialized shared state")

    # Training loop
    train_it = enumerate(train_loader)
    try:
        it = 0
        num_syncs = 0
        while True:
            it += 1
            if it > 1:
                print(f"(it={it}) update_topology()")
                while True:
                    try:
                        communicator.update_topology()
                        break
                    except PCCLError as e:
                        print(f"(it={it}) update_topology() failed: {e}; retrying...")
                        continue

            world_size = communicator.get_attribute(pccl.Attribute.GLOBAL_WORLD_SIZE)

            if world_size > 1:
                while True:
                    try:
                        communicator.optimize_topology()
                        break
                    except pccl.PCCLError as e:
                        print(f"(it={it}) OptimizeTopology failed => {e}. Retrying...")
                        time.sleep(0.1)
                world_size = communicator.get_attribute(pccl.Attribute.GLOBAL_WORLD_SIZE)
    

            if world_size < 2:
                # alone => no ring-based operation => wait
                print("[Peer] alone => sleeping.")
                time.sleep(1)
                continue

            # Perform cuda device synchronization
            # if your shared state partially or fully resides on the GPU we must wait until all currently dispatched kernels have completed
            # to avoid validating or potentially transmitting data that is currently being in-place modified.
            torch.cuda.synchronize(device)
            
            sync_info = communicator.sync_shared_state(shared_state)
            num_syncs += 1
            if shared_state.revision >= max_steps:
                print(f"(it={it}) Training completed")
                break

            assert sync_info is not None
            if num_syncs > 1:
                assert sync_info.rx_bytes == 0

            try:
                batch_idx, (images, labels) = next(train_it)
            except StopIteration:
                print(f"(it={it}) End of epoch")
                train_it = enumerate(train_loader)
                batch_idx, (images, labels) = next(train_it)

            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            # set_to_none=False is required, otherwise shared state references will become invalid!
            optimizer.zero_grad(set_to_none=False)
            loss.backward()

            # collect gradients in one contiguous tensor
            grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])

            while world_size > 1:
                print(f"(it={it}) all_reduce_async()")
                op_desc = ReduceOperandDescriptor(
                    datatype=DataType.FLOAT,
                    distribution_hint=DistributionHint.NORMAL
                )
                quant_desc = QuantizationOptions(
                    quantized_datatype=DataType.FLOAT,
                    algorithm=pccl.QuantizationAlgorithm.NONE
                )
                handle = communicator.all_reduce_async(grads, grads, operand_descriptor=op_desc,
                                                       quantization_options=quant_desc, op=ReduceOp.SUM)
                is_success, status, info = handle.wait()
                world_size = communicator.get_attribute(pccl.Attribute.GLOBAL_WORLD_SIZE)
                if not is_success:
                    print(f"(it={it}) all_reduce_async() failed: {status}; retrying...")
                    continue
                assert info is not None
                print(
                    f"(it={it}) Reduce completed RX: {info.rx_bytes}, TX: {info.tx_bytes}; world_size: {info.world_size}")
                break

            # scatter gradients back to model parameters
            offset = 0
            for p in model.parameters():
                if p.grad is None:
                    continue
                numel = p.numel()
                p.grad.data.copy_(grads[offset:offset + numel].view_as(p.grad))
                offset += numel

            # Update parameters
            optimizer.step()
            shared_state.revision += 1

            if shared_state.revision % 5 == 0:
                print(f"(it={it}) loss: {loss.item()}, revision: {shared_state.revision}")

    except Exception as e:
        print(f"Training aborted with exception: {e}")
        exit(1)

    # Test the model
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Test Accuracy: {100 * correct / total:.2f}%")

    # Save the model
    torch.save(model.state_dict(), "mnist_model.pth")
    print("Model saved to mnist_model.pth")


if __name__ == '__main__':
    main()
```

### Concurrent All-Reduce with Retry

To safely recover from failures when using multiple concurrent collective operations, a scheme such as the following must be employed
to recover safely:

```python
import torch
from pccl import Communicator, ReduceOp, Attribute, ReduceOperandDescriptor, DataType, DistributionHint, QuantizationOptions, QuantizationAlgorithm, ReduceDescriptor, ReduceOpDescriptor, PCCLError

def all_reduce_multiple_with_retry(communicator: Communicator,
                                   tensors: list[torch.Tensor],
                                   op: ReduceOp,
                                   max_in_flight: int = 16):
    """
    Launches concurrent all-reduce operations on a list of tensors,
    waits for them all, and retries if a peer fails or the world size changes.
    Will attempt to target :param max_in_flight: concurrent all-reduce operations.
    The more similar your tensors are in size, the better this in flight system will work.
    """
    descriptors = []
    tag = 0
    for tensor in tensors:
        reduce_op_descriptor = ReduceOpDescriptor.from_torch(
            send=tensor,
            recv=tensor,
            reduce_descriptor=ReduceDescriptor(
                count=tensor.numel(),
                op=op,
                tag=tag,
                operand_descriptor=ReduceOperandDescriptor(
                    datatype=DataType.FLOAT,
                    distribution_hint=DistributionHint.NORMAL
                ),
                quantization_options=QuantizationOptions(
                    quantized_datatype=DataType.FLOAT,
                    algorithm=QuantizationAlgorithm.NONE
                )
            )
        )
        descriptors.append(reduce_op_descriptor)
        tag += 1
    try:
        info = communicator.all_reduce_multiple_with_retry(descriptors, max_in_flight=max_in_flight)
        return True, info.tx_bytes, info.rx_bytes
    except PCCLError:
        return False, 0, 0

def all_reduce_multiple_with_retry__equivalent(communicator: Communicator,
                                   tensors: list[torch.Tensor],
                                   op: ReduceOp,
                                   max_in_flight: int = 16):
    """
    The following function is equivalent to the above, but uses a more manual approach. (Not recommended for production)
    """
    world_size = communicator.get_attribute(Attribute.GLOBAL_WORLD_SIZE)

    total_tx = 0
    total_rx = 0

    def launch_all_reduce(x: torch.Tensor, tag: int):
        op_desc = ReduceOperandDescriptor(
            datatype=DataType.FLOAT,
            distribution_hint=DistributionHint.NORMAL
        )
        quant_desc = QuantizationOptions(
            quantized_datatype=DataType.FLOAT,
            algorithm=QuantizationAlgorithm.NONE
        )
        return communicator.all_reduce_async(
            x, x,
            operand_descriptor=op_desc,
            quantization_options=quant_desc,
            op=op,
            tag=tag
        )

    handles = [None for _ in range(len(tensors))]
    done_handles = set()

    in_flight = 0
    for tensor_index in range(len(tensors)):
        dst_tensor = tensors[tensor_index]

        if in_flight >= max_in_flight:
            break

        handles[tensor_index] = launch_all_reduce(
            dst_tensor,
            tensor_index
        )
        in_flight += 1

    while world_size > 1:
        all_done = True
        for tensor_index in range(len(tensors)):
            handle = handles[tensor_index]
            dst_tensor = tensors[tensor_index]

            if handle is None:
                if tensor_index in done_handles:
                    continue

                if in_flight >= max_in_flight:
                    continue

                handle = handles[tensor_index] = launch_all_reduce(
                    dst_tensor,
                    tensor_index
                )
                in_flight += 1

            is_success, status, info = handle.wait()
            world_size = communicator.get_attribute(Attribute.GLOBAL_WORLD_SIZE)
            if not is_success:
                print(f"Reduce failed: {status}; Starting recovery procedure")
                handles[tensor_index] = None
                # Wait for all ongoing ops to finish or fail before retry
                for j in range(len(tensors)):
                    if j == tensor_index:
                        continue
                    h_j = handles[j]
                    if h_j is not None:
                        s_j, _, _ = h_j.wait()
                        if s_j:
                            done_handles.add(j)
                        in_flight -= 1
                    handles[j] = None
                all_done = False
                break

            # success for this handle
            handles[tensor_index] = None
            done_handles.add(tensor_index)

            total_tx += info.tx_bytes
            total_rx += info.rx_bytes

            in_flight -= 1

        if all_done:
            print(
                f"Reduce completed RX: {total_rx}, TX: {total_tx}; world_size: {world_size}")
            break

    if world_size == 1:
        print(f"All peers have left except this peer. All reduce will do nothing.")
        # If we are alone, just finalize all handles and return
        for h in handles:
            if h is not None:
                h.wait()
        return False

    return True
```