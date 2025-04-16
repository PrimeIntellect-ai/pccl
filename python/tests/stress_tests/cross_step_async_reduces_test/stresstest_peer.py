import threading
from time import sleep
import os
import logging
import torch
from pccl import SharedState, TensorInfo, Communicator, Attribute, ReduceOp, QuantizationOptions, DataType, \
    QuantizationAlgorithm, PCCLError, ReduceOperandDescriptor, DistributionHint, AsyncReduceHandle, ReduceOpDescriptor, \
    ReduceDescriptor
from typing_extensions import Optional, List

HOST: str = '127.0.0.1:48148'
STEPS: int = 1000
PEERS: int = 1
NUM_ELEMENTS: int = 1024 * 1024
RANK: int = int(os.getenv('RANK', "0"))

logging.basicConfig(level=logging.INFO)


def all_reduce_multiple_with_retry(communicator: Communicator,
                                   tensors: list[torch.Tensor],
                                   op: ReduceOp,
                                   max_in_flight: int = 8):
    """
    Launches concurrent all-reduce operations on a list of tensors,
    waits for them all, and retries if a peer fails or the world size changes.
    """
    world_size = communicator.get_attribute(Attribute.GLOBAL_WORLD_SIZE)

    total_tx = 0
    total_rx = 0

    def launch_all_reduce(x: torch.Tensor, tag: int):
        op_desc = ReduceOperandDescriptor(
            datatype=DataType.FLOAT,
            distribution_hint=DistributionHint.NORMAL
        )
        # Example uses min-max quantization to demonstrate concurrency
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
                print(f"(RANK={RANK}) Reduce failed: {status}; Starting recovery procedure")
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
                f"(RANK={RANK}) Reduce completed RX: {total_rx}, TX: {total_tx}; world_size: {world_size}")
            break

    if world_size == 1:
        print(f"(RANK={RANK}) All peers have left except this peer. All reduce will do nothing.")
        # If we are alone, just finalize all handles and return
        for h in handles:
            if h is not None:
                h.wait()
        return False

    return True


def main():
    logging.info(f"(RANK={RANK}) Starting peer node connecting to {HOST}")

    # Create a weight tensor
    weights: torch.Tensor = torch.rand(NUM_ELEMENTS, dtype=torch.float32)

    # Create shared state with tensor infos
    shared_state: SharedState = SharedState([
        TensorInfo.from_torch(weights, 'weights')
    ])
    shared_state.revision = 0

    # Create a communicator and connect to the master node
    communicator: Communicator = Communicator(HOST, 0)
    communicator.connect(n_attempts=15)
    logging.info(f"(RANK={RANK}) Connected to the master node")

    reduce_thread: Optional[threading.Thread] = None

    n_performed_steps = 0
    it = 0
    world_size: int = communicator.get_attribute(Attribute.GLOBAL_WORLD_SIZE)

    # Create fake gradients
    gradients = [torch.randn(128, 128, dtype=torch.float32) for _ in range(10)]

    while True:
        # do step
        topology_updated = False
        if it == 0:
            # Assume that the topology was updated when we just freshly joined the run
            topology_updated = True

        if it > 0:
            for retry in range(10):
                try:
                    logging.info(f"(RANK={RANK}, it={it}) update_topology()")
                    peers_pending = communicator.are_peers_pending()
                    if peers_pending:
                        if reduce_thread is not None:
                            logging.info("(RANK={RANK}) Waiting for reduce_thread to finish before updating topology...")
                            reduce_thread.join()
                        communicator.update_topology()
                        topology_updated = True
                    break
                except PCCLError as ex:
                    if retry == 10:
                        # The peer was likely kicked; this can happen be collateral damage in rare cases when lots of
                        # people die at the same time, p2p connections cannot be established and the master thinks
                        # the peer cannot communicate with enough peers to build a tour. This is unresolvable because
                        # it might just be true that the peer can indeed just not talk to those peers.
                        # If they had disconnected from the master too, they would already be considered removed from the
                        # run. But if that happens just a tick late, this unlucky situation can happen.
                        print(
                            "Exiting because peer was likely kicked. This doesn't necessary entail a failure of the stress test, as long as the shared state revision is not lost and new peers continue on.")
                        exit(0)
                    print(f"(RANK={RANK}, it={it}) update_topology() failed: {ex}; retrying...")
                    continue

            world_size = communicator.get_attribute(Attribute.GLOBAL_WORLD_SIZE)

        if world_size > 1 and topology_updated:
            while True:
                try:
                    logging.info(f"(RANK={RANK}, it={it}) optimize_topology()")
                    # communicator.optimize_topology()
                    break
                except PCCLError:
                    sleep(0.1)
                    logging.info(f"(RANK={RANK}, it={it}) optimize_topology failed; retrying...")
                    continue
        world_size = communicator.get_attribute(Attribute.GLOBAL_WORLD_SIZE)

        if world_size < 2:
            logging.info(f"(RANK={RANK}, it={it}) waiting...")
            sleep(1)
            it += 1
            continue

        if topology_updated:
            logging.info(f"(RANK={RANK}, it={it}) sync_shared_state()")
            info = communicator.sync_shared_state(shared_state)
            shared_state.revision += 1
            assert info is not None
            print(f"(RANK={RANK}, it={it}) tx_bytes={info.tx_bytes}, rx_bytes={info.rx_bytes}")

        # Re-init gradients randomly
        for i in range(len(gradients)):
            torch.randn(out=gradients[i], size=gradients[i].shape)

        def reduce_thread_fn_2():
            descriptors = []
            tag = 0
            for grad in gradients:
                reduce_op_descriptor = ReduceOpDescriptor.from_torch(
                    send=grad,
                    recv=grad,
                    reduce_descriptor=ReduceDescriptor(
                        count=grad.numel(),
                        op=ReduceOp.SUM,
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

            reduce_info = communicator.all_reduce_multiple_with_retry(descriptors, max_in_flight=8)
            print(
                f"(RANK={RANK}) Reduce completed RX: {reduce_info.rx_bytes}, TX: {reduce_info.tx_bytes}; world_size: {world_size}")

        def reduce_thread_fn():
            all_reduce_multiple_with_retry(communicator, gradients, ReduceOp.SUM, max_in_flight=8)

        if reduce_thread is not None:
            reduce_thread.join()

        reduce_thread = threading.Thread(target=reduce_thread_fn_2, name="ReduceThread")
        reduce_thread.start()

        n_performed_steps += 1
        it += 1

    logging.info(f"(RANK={RANK}) Finished")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"(RANK={RANK}) failed with: {e}")
        raise e
