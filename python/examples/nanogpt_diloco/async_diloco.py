import os
import threading
import time
import math
import pickle
import json
import argparse
import zlib
from contextlib import nullcontext

import numpy as np
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

import pccl
from pccl import (
    Communicator,
    Attribute,
    SharedState,
    TensorInfo,
    DataType,
    DistributionHint,
    QuantizationOptions,
    QuantizationAlgorithm,
    ReduceOp,
    ReduceOperandDescriptor,
    PCCLError, ReduceOpDescriptor, ReduceDescriptor
)
from typing_extensions import Optional

from model import GPTConfig, GPT
from python.examples.nanogptddp.profiler import Profiler, ProfilerCollection


def all_reduce_multiple_with_retry(communicator: Communicator,
                                   tensors: list[torch.Tensor],
                                   op: ReduceOp,
                                   max_in_flight: int = 16):
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
                    quantized_datatype=DataType.UINT8,
                    algorithm=QuantizationAlgorithm.MIN_MAX
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

def get_batch(split, config, device_type, device, train: bool):
    """
    Poor man's data loader. Each call randomly selects `batch_size` samples
    from train or val memmapped .bin files and returns them as x, y tensors.
    """
    data_dir = os.path.join('data', config["dataset"])
    block_size = config["block_size"]
    batch_size = config["batch_size"] if train else config["val_batch_size"]

    # We recreate np.memmap every call to avoid memory leaks on repeated usage
    data_path = os.path.join(data_dir, 'train.bin' if split == 'train' else 'val.bin')
    data = np.memmap(data_path, dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([
        torch.from_numpy((data[i: i + block_size]).astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy((data[i + 1: i + 1 + block_size]).astype(np.int64)) for i in ix
    ])

    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss(model, ctx, config, get_batch_fn, device_type, device):
    """
    Estimate the mean loss on train and val sets by running `eval_iters` random mini-batches.
    """
    eval_iters = config["eval_iters"]
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            x, y = get_batch_fn(split, config, device_type, device, False)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it, config):
    """
    Learning rate decay scheduler (cosine with warmup).
    If decay_lr is False, just return the original learning_rate.
    """
    if not config["decay_lr"]:
        return config["learning_rate"]

    learning_rate = config["learning_rate"]
    warmup_iters = config["warmup_iters"] // config["inner_steps"]
    lr_decay_iters = config["lr_decay_iters"] // config["inner_steps"]
    min_lr = config["min_lr"]

    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    decay_ratio = max(0, min(1, decay_ratio))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


EXPORT_PROFILER_VIDEO = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="default_config.json",
        help="Path to JSON configuration file."
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 1) Load configuration from JSON
    # -------------------------------------------------------------------------
    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # If BF16 is available, override float16 with bf16
    if (config["dtype"] == "float16") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        config["dtype"] = "bfloat16"

    # -------------------------------------------------------------------------
    # 2) Initialize DDP if needed (we won't use it for DiLoCo logic, but keep for environment)
    # -------------------------------------------------------------------------
    ddp = int(os.environ.get('RANK', -1)) != -1  # check if distributed run
    if ddp:
        init_process_group(backend=config["backend"])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = (ddp_rank == 0)
        # scale down gradient_accumulation_steps if you wish (not used in DiLoCo)
        gas = config["gradient_accumulation_steps"]
        if gas % ddp_world_size != 0:
            raise ValueError(
                "gradient_accumulation_steps must be divisible by WORLD_SIZE in the old logic. For DiLoCo, it's irrelevant.")
        config["gradient_accumulation_steps"] = gas // ddp_world_size
    else:
        master_process = config["is_master_process"]
        ddp_world_size = 1
        device = config["device"]

    tokens_per_iter = (
            config["batch_size"] * config["block_size"] * ddp_world_size
    )
    print(f"tokens per iteration (outer step) will be: {tokens_per_iter:,} (per local update)")

    # -------------------------------------------------------------------------
    # 3) Setup environment, seeds, output directory
    # -------------------------------------------------------------------------
    if master_process:
        os.makedirs(config["out_dir"], exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }[config["dtype"]]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # -------------------------------------------------------------------------
    # 4) Possibly read vocab size from meta
    # -------------------------------------------------------------------------
    data_dir = os.path.join('data', config["dataset"])
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # -------------------------------------------------------------------------
    # 5) Create or resume local model
    # -------------------------------------------------------------------------
    model_args = dict(
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        block_size=config["block_size"],
        bias=config["bias"],
        vocab_size=None,
        dropout=config["dropout"]
    )

    iter_num = 0
    best_val_loss = 1e9
    init_from = config["init_from"]
    checkpoint = None

    if init_from == 'scratch':
        print("Initializing a new model from scratch.")
        if meta_vocab_size is None:
            print("Defaulting to GPT-2 vocab_size of 50304.")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

    elif init_from == 'resume':
        print(f"Resuming training from {config['out_dir']}")
        ckpt_path = os.path.join(config["out_dir"], 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        override_args = dict(dropout=config["dropout"])
        model = GPT.from_pretrained(init_from, override_args)
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
    else:
        raise ValueError(f"Unknown init_from: {init_from}")

    # Possibly reduce block_size
    if config["block_size"] < model.config.block_size:
        model.crop_block_size(config["block_size"])
        model_args['block_size'] = config["block_size"]

    model.to(device)

    # -------------------------------------------------------------------------
    # 6) Set up the DiLoCo approach:
    #    - "inner_optimizer": local updates
    #    - "outer_params" & "outer_optimizer": aggregated updates
    # -------------------------------------------------------------------------
    # Inner optimizer (same Adam config from the original)
    inner_optimizer = model.configure_optimizers(
        config["weight_decay"],
        config["learning_rate"],
        (config["beta1"], config["beta2"]),
        device_type
    )
    if init_from == 'resume':
        inner_optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint  # free memory

    if config["compile"]:
        print("Compiling the model... (PyTorch 2.0)")
        model = torch.compile(model)

    # If ddp is set, just wrap model in DDP for consistent environment usage
    if ddp:
        model = DistributedDataParallel(model, device_ids=[int(device.split(":")[-1])])

    raw_model = model.module if ddp else model

    # Outer parameters / outer optimizer
    outer_params_dict = {}
    for name, local_p in raw_model.named_parameters():
        outer_params_dict[name] = torch.nn.Parameter(local_p.detach().cpu())

    outer_params_list = []
    for name, local_p in raw_model.named_parameters():
        outer_params_list.append(outer_params_dict[name])

    # Let's use SGD w/ momentum for aggregator
    outer_optimizer = optim.SGD(
        outer_params_list,
        lr=config["outer_learning_rate"]
    )
    # do a dummy step to initialize outer optimizer state
    for op in outer_params_list:
        op.grad = torch.zeros_like(op)
    outer_optimizer.step()

    # -------------------------------------------------------------------------
    # 7) Initialize PCCL communicator + SharedState for outer parameters
    # -------------------------------------------------------------------------
    communicator = Communicator(config["ccoip_host"], 0)
    communicator.connect(n_attempts=15)
    print("Connected to master via PCCL.")

    # Build the shared state that includes:
    #   - The "outer_params" themselves
    #   - The outer optimizer state (e.g. momentum buffers)
    shared_state_dict = {}
    for name, param in outer_params_dict.items():
        shared_state_dict[name] = param

    # Outer optimizer momentum buffers in shared state
    for name, outer_p in outer_params_dict.items():
        # momentum_buffer is allocated after the first step
        state = outer_optimizer.state[outer_p]
        momentum_buf = state.get("momentum_buffer", None)
        if momentum_buf is not None:
            shared_state_dict[f"{name}_momentum_buffer"] = momentum_buf

    iter_num = torch.tensor(iter_num, dtype=torch.int64, device='cpu')
    shared_state_dict['iter_num'] = iter_num

    entries = [
        TensorInfo.from_torch(tensor, name, allow_content_inequality=False)
        for name, tensor in shared_state_dict.items()
    ]
    shared_state = SharedState(entries)
    shared_state.revision = 0

    # If wandb logging is enabled
    if config["wandb_log"] and master_process:
        import wandb
        wandb.init(
            project=config["wandb_project"],
            config=config
        )

    # -------------------------------------------------------------------------
    # 8) Training loop (DiLoCo version)
    # -------------------------------------------------------------------------
    profiler_collection = ProfilerCollection()
    local_iter_num = 0
    num_syncs = 0
    running_mfu = -1.0

    world_size: int = communicator.get_attribute(Attribute.GLOBAL_WORLD_SIZE)

    all_reduce_thread: Optional[threading.Thread] = None

    def run_shared_state_sync(late_joiner: bool = False):
        nonlocal iter_num, num_syncs
        with profiler.session("pccl::sync_shared_state"):
            sync_info = communicator.sync_shared_state(shared_state)
            shared_state.revision += 1
            print(f"sync_info tx_bytes: {sync_info.tx_bytes}, rx_bytes: {sync_info.rx_bytes}")
            num_syncs += 1
            if num_syncs > 1 and not late_joiner:
                assert sync_info.rx_bytes == 0, "Shared state drifted unexpectedly in peers!"

            # initialize inner state on first sync
            if num_syncs == 1:
                print("Initializing outer state...")
                with torch.no_grad():
                    for inner_param, outer_param in zip(model.parameters(), outer_params_list):
                        inner_param.data.copy_(outer_param)

    last_pseudo_grads = []
    while True:
        local_iter_num += 1
        t0 = time.time()
        profiler = Profiler()

        topology_updated = False
        if local_iter_num == 1:
            # Assume the topology was updated in the first iteration because we just joined and got accepted
            topology_updated = True

        # 1) Possibly update topology / wait for enough peers
        with profiler.session("pccl::update_topology"):
            if local_iter_num > 1 or world_size == 1:
                while True:
                    try:
                        if communicator.are_peers_pending():
                            print("peers pending; awaiting concurrent collective ops and accepting new peers...")
                            if all_reduce_thread is not None:
                                all_reduce_thread.join()
                            communicator.update_topology()
                            topology_updated = True
                        break
                    except PCCLError as e:
                        print(f"update_topology() failed => {e}, retrying...")
                        time.sleep(1)

        world_size = communicator.get_attribute(Attribute.GLOBAL_WORLD_SIZE)
        if world_size < 2:
            print("Waiting for more workers to join...")
            time.sleep(1)
            continue

        # 2) Attempt topology optimization
        if topology_updated and world_size > 1:
            while True:
                try:
                    # communicator.optimize_topology()
                    break
                except pccl.PCCLError as e:
                    print(f"optimize_topology() failed => {e}, retrying...")
                    time.sleep(1)

        # 3) Sync shared state => ensures we have the same aggregator (outer) parameters
        if topology_updated:
            run_shared_state_sync()

        # Set learning rate for both inner and outer optimizers
        lr = get_lr(iter_num.item(), config)
        for param_group in inner_optimizer.param_groups:
            param_group['lr'] = lr

        # 4) Evaluate / checkpoint if needed
        #    (We do this before local steps, so it sees the "current" global model.)
        if iter_num.item() % (config["eval_interval"] // config["inner_steps"]) == 0 and master_process:
            with profiler.session("estimate_loss"):
                losses = estimate_loss(raw_model, ctx, config,
                                       get_batch, device_type, device)
            print(f"step {iter_num.item()}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if config["wandb_log"]:
                import wandb
                wandb.log({
                    "iter": iter_num.item(),
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu * 100,
                })

            if losses['val'] < best_val_loss or config["always_save_checkpoint"]:
                best_val_loss = losses['val']
                if iter_num.item() > 0:
                    ckpt = {
                        'model': raw_model.state_dict(),
                        'optimizer': inner_optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num.item(),
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    ckpt_path = os.path.join(config["out_dir"], 'ckpt.pt')
                    print(f"Saving checkpoint to {ckpt_path}")
                    with profiler.session("save_checkpoint"):
                        torch.save(ckpt, ckpt_path)

        if iter_num == 0 and config["eval_only"]:
            break

        # 5) Perform local (inner) steps
        #    We do config["inner_steps"] steps of local training
        #    on random mini-batches
        with profiler.session("inner_steps"):
            raw_model.train()
            for s in range(config["inner_steps"]):
                avg_loss = 0
                for micro_step in range(config["gradient_accumulation_steps"]):
                    x, y = get_batch('train', config, device_type, device, True)
                    with ctx:
                        logits, loss = model(x, y)

                    loss.backward()
                    avg_loss += loss.item()

                if config["grad_clip"] != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                inner_optimizer.step()
                inner_optimizer.zero_grad(set_to_none=False)
                avg_loss /= config["gradient_accumulation_steps"]
                print(f"Inner step {s}: loss {avg_loss}")
                if config["wandb_log"] and master_process:
                    import wandb
                    wandb.log({
                        "iter": iter_num,
                        "train/inner/loss": avg_loss
                    })
                    pass

            # await previous all reduce, if one exists
            can_outer_step = False
            if all_reduce_thread is not None:
                all_reduce_thread.join()
                can_outer_step = True

                # populate outer param grads with last pseudo-gradients set by thread
                for pseudo_grad, outer_p in zip(last_pseudo_grads, outer_params_list):
                    outer_p.grad = pseudo_grad

            # Compute current pseudo grads as difference between outer and inner state.
            # Inner state is advanced by inner steps, outer state is unchanged
            outer_grads = []
            for (local_param, outer_param) in zip(raw_model.parameters(), outer_params_list):
                pseudo_grad = outer_param - local_param.data.to('cpu')
                outer_grads.append(pseudo_grad)

            if can_outer_step:
                outer_optimizer.step()  # Note that there is no zero-grad because grads get re-instantiated every step

                # (d) Copy aggregator result into local model
                with torch.no_grad():
                    for (local_p, outer_p) in zip(raw_model.parameters(), outer_params_list):
                        local_p.copy_(outer_p, non_blocking=True)

                if topology_updated and iter_num > 0:
                    # If the topology was updated and iter_num is > 0
                    # then a new peer just joined the run with needs to be properly inserted into
                    # the N-1 async pipeline.
                    # To do this we first initially sync the weights such that the peer can
                    # start computing the current step like the pre-existing peers, however
                    # the newly joined peer cannot be "retroactively inserted" into
                    # the N-1 async reduce that was started last step.
                    # So it needs to "eavesdrop" on the result that the other peers are about to compute
                    # with a second shared state re-transmission.
                    # Hence, both pre-existing peers and newly joined peer(s) have to perform shared state
                    # synchronization.
                    # The pre-existing peers first apply the outer optimizer and THEN call run_shared_state_sync
                    # because the new peer(s) need to obtain the shared state as it is after the all reduce
                    # is applied that they were not part of.
                    print(
                        "Topology updated mid run; re-running shared state synchronization to properly insert new peer...")
                    run_shared_state_sync()
            else:
                if topology_updated and iter_num > 0:
                    # If the topology was updated and iter_num is > 0 and can_outer_step is False,
                    # then WE are the joining peer to an ongoing run.
                    # In this case, we have to obtain the shared state from the pre-existing peers.
                    # We obtain the shared state first and then simply copy it into the inner model afterwards.
                    # Also: late_joiner here means that we tolerate actually receiving bytes here despite that this is the second sync that was performed.
                    # This is necessary for the pipeline insertion algorithm to function
                    run_shared_state_sync(late_joiner=True)

                # This is the boostrap for the 1-step behind asynchronous training step.
                # Reset the inner state here to be equal to the unmodified outer state.
                # This essentially resets the model back to initialization state.
                # Why do this?
                # a) because the next shared state sync needs to see all outer states as equal.
                # We haven't communicated yet, so we have by definition diverged.
                # But we will hide this for now.
                # b) what we are accomplishing here is as follows:
                # We know that the pseudo-grads constitute a valid update to the weights
                # to decrease the loss when applied to the initial model state.
                # These changes will be applied in the next loop iteration.
                # We will hide the communication with compute of the next iteration.
                # Afterward, we will apply said delta to the still initial weights.
                # At this stage, we haven't done anything questionable at all.
                # We have applied a valid update to exactly the base weights they were grads for.
                # However, now in the next outer step, the reduce of the pseudo-gradients of step two is awaited
                # and these are updates from initial weights also - just derived from different input data.
                # We have already moved on from the initial weights
                # at this point. And yet, we still apply them. This is the 1-step behind assertion
                # that we make that it is reasonable to still apply these gradients, even though they
                # are slightly outdated. From then onwards, outer step updates are always one step behind.
                with torch.no_grad():
                    for (local_p, outer_p) in zip(raw_model.parameters(), outer_params_list):
                        local_p.copy_(outer_p, non_blocking=True)

            def run_all_reduce():
                nonlocal last_pseudo_grads
                last_pseudo_grads = outer_grads.copy()
                start_time = time.time()
                all_reduce_multiple_with_retry(
                    communicator,
                    last_pseudo_grads,
                    ReduceOp.AVG
                )
                end_time = time.time()
                print(f"All-Reduce took {end_time - start_time} seconds")

            all_reduce_thread = threading.Thread(target=run_all_reduce, name="ReduceThread")

            # NOTE: no zero-grad on outer grads, as they continue to get referenced by this thread.
            all_reduce_thread.start()

        # 7) Some logging / housekeeping
        t1 = time.time()
        dt = t1 - t0

        # We'll do the MFU calculation for reference
        tokens_this_iter = config["batch_size"] * config["gradient_accumulation_steps"] * config["inner_steps"]
        raw_model.eval()  # to call estimate_mfu
        mfu = raw_model.estimate_mfu(tokens_this_iter, dt)
        raw_model.train()
        running_mfu = mfu if running_mfu < 0 else 0.9 * running_mfu + 0.1 * mfu

        if iter_num.item() % config["log_interval"] == 0 and master_process:
            print(
                f"iter {iter_num}: time {dt * 1000:.2f}ms, local loss {loss.item():.4f}, mfu {running_mfu * 100:.2f}%")

        iter_num += 1

        # 8) Termination condition
        if iter_num.item() > config["max_iters"] // config["inner_steps"]:
            break

        profiler.print_report()
        if EXPORT_PROFILER_VIDEO:
            profiler_collection.add_profiler(profiler, f"Step {iter_num.item()}")
            if iter_num % 100 == 0:
                outpath = f'timeline_{iter_num}.mp4'
                profiler_collection.render_as_video(outpath, fps=15)
                print(f"Done rendering profiler video to {outpath}")

    # -------------------------------------------------------------------------
    # 9) Cleanup
    # -------------------------------------------------------------------------
    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
