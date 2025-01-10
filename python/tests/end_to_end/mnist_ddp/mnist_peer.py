import os
import signal
import time
import zlib
from time import sleep
from typing import List

import pccl
from pccl import Communicator, SharedState, TensorInfo, ReduceInfo, ReduceOp, Attribute, PCCLError
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def human_readable_bytes(size: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            break
        size /= 1024.0
    return f"{size:.2f} {unit}"


def compute_crc32(tensor: torch.Tensor) -> int:
    tensor_cpu = tensor.detach().cpu()
    tensor_contiguous = tensor_cpu.contiguous()
    tensor_np = tensor_contiguous.numpy()
    tensor_bytes = tensor_np.tobytes()
    checksum = zlib.crc32(tensor_bytes)
    return checksum


# Set device
device = torch.device("cpu")

# Define hyperparameters
input_size = 28 * 28  # MNIST images are 28x28
# hidden_sizes = [1024, 4096, 4096, 4096, 4096, 1024]
hidden_sizes = [128]
num_classes = 10  # Digits 0-9
batch_size = 64
learning_rate = 0.001
max_steps = os.getenv('IS_CI', '0') == '1' and 2048 or 5000

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


LOG_DEBUG = True


def log_debug(msg: str):
    if LOG_DEBUG:
        print(msg)


def log_info(msg: str):
    print(msg)


HOST: str = '127.0.0.1:48148'
RANK: int = int(os.getenv('RANK', "0"))
CREATE_RANK_0_REV_50: int = int(os.getenv('CREATE_RANK_0_REV_50', "0"))


def main():
    model = NeuralNet(input_size, hidden_sizes, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # communicator
    communicator: Communicator = Communicator(HOST, 0)
    n_attempts = 5
    for attempt in range(n_attempts):
        try:
            communicator.connect()
            break
        except PCCLError as e:
            print(
                f"(RANK={RANK}) Failed to connect to the master node: {e}; (Attempt {attempt + 1}/{n_attempts})")

            sleep(1)
    else:
        assert False, f"(RANK={RANK}) Failed to connect to the master node"
    log_info(f"(RANK={RANK}) Connected to the master node; PID={os.getpid()}")

    # perform a dummy forward pass to initialize the optimizer state
    for p in model.parameters():
        p.grad = torch.zeros_like(p)  # set all gradients to zero
    optimizer.step()
    print(f"(RANK={RANK}) Initialized optimizer state")

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
    print(f"(RANK={RANK}) Initialized shared state")

    # Training loop
    train_it = enumerate(train_loader)
    try:
        i = 0
        while True:
            i += 1
            if i > 1:
                log_debug(f"(RANK={RANK}, it={i}) update_topology()")
                communicator.update_topology()

            world_size = communicator.current_world_size()

            if world_size < 2:
                sleep(1)
                continue

            log_debug(f"(RANK={RANK}, it={i}, ws={world_size}) sync_shared_state()")
            sync_info = communicator.sync_shared_state(shared_state)
            log_debug(
                f"(RANK={RANK}, it={i}) shared_state.revision: {shared_state.revision}, sync_info (tx_bytes={sync_info.tx_bytes}, rx_bytes={sync_info.rx_bytes})")
            if shared_state.revision >= max_steps:
                log_debug(f"(RANK={RANK}, it={i}) Training completed")
                break

            try:
                batch_idx, (images, labels) = next(train_it)
            except StopIteration:
                log_debug(f"(RANK={RANK}, it={i}) End of epoch")
                train_it = enumerate(train_loader)
                batch_idx, (images, labels) = next(train_it)

            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad(set_to_none=False)
            loss.backward()

            # collect gradients in one contiguous tensor
            grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])

            while True:
                log_debug(f"(RANK={RANK}, it={i}) all_reduce_async()")
                handle = communicator.all_reduce_async(grads, grads, numel=grads.numel(), op=ReduceOp.SUM)
                is_success, status, info = handle.wait()
                if not is_success:
                    log_debug(f"(RANK={RANK}, it={i}) all_reduce_async() failed: {status}; retrying...")
                    continue
                assert info is not None
                log_debug(f"(RANK={RANK}, it={i}) Reduce completed RX: {info.rx_bytes}, TX: {info.tx_bytes}; world_size: {info.world_size}")
                break

            # print hash of the gradients tensor content
            log_debug(f"(RANK={RANK}, it={i}) grads hash: {compute_crc32(grads)}")

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

            log_debug(f"(RANK={RANK}, it={i}) loss: {loss.item()}, revision: {shared_state.revision}")

            if shared_state.revision % 5 == 0:
                log_info(f"(RANK={RANK}, it={i}) loss: {loss.item()}, revision: {shared_state.revision}")

            if RANK == 0 and CREATE_RANK_0_REV_50 == 1 and shared_state.revision == 50:
                rev50_signal_file = os.path.join(os.path.dirname(__file__), 'RANK_0_REV_50')
                with open(rev50_signal_file, 'w') as f:
                    f.write('')

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

        log_info(f"Test Accuracy: {100 * correct / total:.2f}%")

    # Save the model
    torch.save(model.state_dict(), "mnist_model.pth")
    log_info("Model saved to mnist_model.pth")


if __name__ == '__main__':
    main()
