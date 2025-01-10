from time import sleep
import os
import logging
from pccl import *

HOST: str = '127.0.0.1:48148'
STEPS: int = 10000
WEIGHT_N: int = 1024
PEERS: int = 1
NUM_ELEMENTS: int = 1024
RANK: int = int(os.getenv('RANK', "0"))


def main():
    logging.info(f"(RANK={RANK}) Starting peer node connecting to {HOST}")

    # Create a weight tensor
    weights: torch.Tensor = torch.rand(WEIGHT_N, dtype=torch.float32)

    # Create shared state with tensor infos
    shared_state: SharedState = SharedState([
        TensorInfo.from_torch(weights, 'weights')
    ])

    # Create a communicator and connect to the master node
    communicator: Communicator = Communicator(HOST, 0)

    n_attempts = 5
    for attempt in range(n_attempts):
        try:
            communicator.connect()
            break
        except PCCLError as e:
            logging.error(f"(RANK={RANK}) Failed to connect to the master node: {e}; (Attempt {attempt + 1}/{n_attempts})")

            sleep(1)
    else:
        assert False, f"(RANK={RANK}) Failed to connect to the master node"
    logging.info(f"(RANK={RANK}) Connected to the master node")

    world_size: int = communicator.current_world_size()

    n_performed_steps = 0
    i = 0
    while n_performed_steps < STEPS:
        if i > 0 or world_size == 1:
            logging.info(f"(RANK={RANK}, it={i}) update_topology()")
            communicator.update_topology()
        world_size = communicator.current_world_size()

        if world_size < 2:
            sleep(1)
            continue

        logging.info(f"(RANK={RANK}, it={i}) sync_shared_state()")
        communicator.sync_shared_state(shared_state)

        if shared_state.revision == STEPS:
            break

        # Create gradients tensors
        grad: torch.Tensor = torch.rand(NUM_ELEMENTS, dtype=torch.float32)
        while True:
            logging.info(f"(RANK={RANK}, it={i}) all_reduce_async()")
            handle = communicator.all_reduce_async(grad, weights, numel=NUM_ELEMENTS, op=ReduceOp.SUM)
            is_success, status, info = handle.wait()
            assert is_success == True, f"All reduce failed with stats: {status}"
            assert info is not None
            logging.info(f"((RANK={RANK}, it={i}) Reduce completed RX: {info.rx_bytes}, TX: {info.tx_bytes}")
            break

        shared_state.revision += 1
        i += 1

    logging.info(f"(RANK={RANK}) Finished")


if __name__ == '__main__':
    try:
        main()
    except (Exception, AssertionError) as e:
        print(f"(RANK={RANK}) failed with: {e}")
        raise e
