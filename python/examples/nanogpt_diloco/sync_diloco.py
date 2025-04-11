import os
import time
import math
import pickle
import json
import argparse
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
    ReduceOpDescriptor,
    ReduceOperandDescriptor,
    PCCLError, ReduceDescriptor
)

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


def get_batch(split, config, device_type, device):
    """
    Poor man's data loader. Each call randomly selects `batch_size` samples
    from train or val memmapped .bin files and returns them as x, y tensors.
    """
    data_dir = os.path.join('data', config["dataset"])
    block_size = config["block_size"]
    batch_size = config["batch_size"]

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
            x, y = get_batch_fn(split, config, device_type, device)
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
    start_iter_num = 0
    if init_from == 'resume':
        inner_optimizer.load_state_dict(checkpoint['optimizer'])
        jj1 = checkpoint['iter_num']
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
    #   - The inner parameters
    #   - The outer optimizer state (e.g. momentum buffers)
    shared_state_dict = {}
    for name, param in model.named_parameters():
        shared_state_dict[name] = param

    # Outer optimizer momentum buffers in shared state
    for name, outer_p in outer_params_dict.items():
        # momentum_buffer is allocated after the first step
        state = outer_optimizer.state[outer_p]
        momentum_buf = state.get("momentum_buffer", None)
        if momentum_buf is not None:
            shared_state_dict[f"{name}_momentum_buffer"] = momentum_buf

    entries = [
        TensorInfo.from_torch(tensor, name, allow_content_inequality=False)
        for name, tensor in shared_state_dict.items()
    ]
    shared_state = SharedState(entries)
    shared_state.revision = start_iter_num
    print("start_iter_num:", start_iter_num)

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

    # For each outer iteration:
    while True:
        local_iter_num += 1
        t0 = time.time()
        profiler = Profiler()

        # 1) Possibly update topology / wait for enough peers
        with profiler.session("pccl::update_topology"):
            if local_iter_num > 1 or world_size == 1:
                while True:
                    try:
                        communicator.update_topology()
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
        if world_size > 1:
            while True:
                try:
                    # communicator.optimize_topology()
                    break
                except pccl.PCCLError as e:
                    print(f"optimize_topology() failed => {e}, retrying...")
                    time.sleep(1)

        # 3) Sync shared state => ensures we have the same aggregator (outer) parameters
        with profiler.session("pccl::sync_shared_state"):
            sync_info = communicator.sync_shared_state(shared_state)
            print(f"sync_info tx_bytes: {sync_info.tx_bytes}, rx_bytes: {sync_info.rx_bytes}")
            iter_num = shared_state.revision
            num_syncs += 1
            if num_syncs > 1:
                assert sync_info.rx_bytes == 0, "Shared state drifted unexpectedly in peers!"

            # initialize outer state on first sync
            if num_syncs == 1:
                print("Initializing outer state...")
                with torch.no_grad():
                    for outer_param, inner_param in zip(outer_params_list, model.parameters()):
                        outer_param.copy_(inner_param.data)

        # Set learning rate for both inner and outer optimizers
        lr = get_lr(iter_num, config)
        for param_group in inner_optimizer.param_groups:
            param_group['lr'] = lr

        # 4) Evaluate / checkpoint if needed
        #    (We do this before local steps, so it sees the "current" global model.)
        if iter_num % (config["eval_interval"] // config["inner_steps"]) == 0 and master_process:
            if iter_num > 0:
                ckpt = {
                    'model': raw_model.state_dict(),
                    'optimizer': inner_optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
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
                    x, y = get_batch('train', config, device_type, device)
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

        with profiler.session("outer_step"):
            outer_grads = []
            for (local_p, outer_p) in zip(raw_model.parameters(), outer_params_list):
                outer_p.grad = outer_p - local_p.data.to('cpu')
                outer_grads.append(outer_p.grad)

            with profiler.session("all_reduce_multiple_with_retry"):
                start_time = time.time()

                all_reduce_success = all_reduce_multiple_with_retry(
                    communicator,
                    outer_grads,
                    ReduceOp.AVG,
                    max_in_flight=128
                )

                end_time = time.time()
                print(f"All-Reduce took {end_time - start_time} seconds")
                if not all_reduce_success:
                    print("All peers left except me... continuing alone.")

            outer_optimizer.step()
            outer_optimizer.zero_grad()

            # (d) Copy aggregator result into local model
            with torch.no_grad():
                for (local_p, outer_p) in zip(raw_model.parameters(), outer_params_list):
                    local_p.copy_(outer_p, non_blocking=True)

        # 7) Some logging / housekeeping
        t1 = time.time()
        dt = t1 - t0

        # We'll do the MFU calculation for reference
        tokens_this_iter = config["batch_size"] * config["gradient_accumulation_steps"] * config["inner_steps"]
        raw_model.eval()  # to call estimate_mfu
        mfu = raw_model.estimate_mfu(tokens_this_iter, dt)
        raw_model.train()
        running_mfu = mfu if running_mfu < 0 else 0.9 * running_mfu + 0.1 * mfu

        if iter_num % config["log_interval"] == 0 and master_process:
            print(
                f"iter {iter_num}: time {dt * 1000:.2f}ms, local loss {loss.item():.4f}, mfu {running_mfu * 100:.2f}%")

        iter_num += 1
        shared_state.revision = iter_num  # keep revision in sync

        # 8) Termination condition
        if iter_num > config["max_iters"] // config["inner_steps"]:
            break

        profiler.print_report()
        if EXPORT_PROFILER_VIDEO:
            profiler_collection.add_profiler(profiler, f"Step {iter_num}")
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
