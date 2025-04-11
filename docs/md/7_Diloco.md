# DiLoCo: Distributed Low-Communication

DiLoCo is an optimization scheme which requires drastically less frequent communication
than naive DDP.
While PCCL is a generalized collective communications library and can thus implement many other optimization schemes,
it was designed with the unique requirements of DiLoCo in mind.

PCCL can implement both synchronous and "streaming" (asynchronous) DiLoCo utilizing PCCL's
async collective primitives paired with its shared state system.

## Synchronous DiLoCo
At a high level, the pseudo-code for implementing synchronous DiLoCo is as follows:
```python
# 1) Build your local model as usual:
local_model = GPT(config)
local_optimizer = AdamW(local_model, config)

# 2) Create separate "outer" parameters on CPU for aggregation:
outer_params = {}
for (name, local_param) in local_model.named_parameters():
    outer_params[name] = new_parameter_on_cpu( local_param.shape )

# 3) Create an "outer" optimizer, e.g. SGD:
outer_optimizer = SGD(outer_params, config)

# 4) Create PCCL communicator and connect:
communicator = Communicator(host=config["ccoip_host"], port=48148)
communicator.connect()

# 5) Build a SharedState containing all relevant parameters (model + any momentum buffers, etc.):
shared_state_dict = collect_all_params_and_buffers(local_model, outer_optimizer)
tensor_info_entries = [TensorInfo.from_torch(tensor, name) for (name, tensor) in shared_state_dict.items()]
shared_state = SharedState(tensor_info_entries)
shared_state.revision = 0  # e.g., the global "iteration number"

# TRAINING LOOP (SIMPLIFIED)
for outer_iter in range(MAX_OUTER_ITERS):

    # (A) Join aggregator "topology" via PCCL
    communicator.update_topology()

    sync_info = communicator.sync_shared_state(shared_state)

    # Copy aggregator data from shared_state into your local model if needed:
    #   e.g., for each "outer_param" in shared_state_dict, copy to local_model
    read_outer_params_into_local(local_model, outer_params)

    # (C) Perform local (inner) steps on random mini-batches
    for _ in range(config["inner_steps"]):
        loss = local_train_step(local_model, local_optimizer, batch_data())

    # (D) Compute difference: (outer_param - local_param)
    #     Then reduce among all workers => aggregator step
    outer_grad_list = []
    for name in outer_params:
        # CPU param minus local GPU param => gradient
        outer_params[name].grad = (outer_params[name] - local_model[name].cpu_data())

        outer_grad_list.append(outer_params[name].grad)

    # Use PCCL all_reduce_multiple_with_retry to average these differences:
    success, tx_bytes, rx_bytes = all_reduce_multiple_with_retry(
        communicator,
        outer_grad_list,
        ReduceOp.AVG
    )

    # Then outer optimizer steps:
    outer_optimizer.step()
    outer_optimizer.zero_grad()

    # Finally copy aggregated outer params back into local model:
    copy_outer_params_to_local(local_model, outer_params)

    # (E) Increment shared_state.revision and loop
    shared_state.revision += 1
    if termination_condition(...):
        break
```

A full example implementing synchronous DiLoCo is available here: https://github.com/PrimeIntellect-ai/pccl/blob/main/python/examples/nanogpt_diloco/sync_diloco.py
A full example implementing synchronous DiLoCo combined with FSDP (Fully Sharded Data Parallel) is available here: https://github.com/PrimeIntellect-ai/pccl/blob/main/python/examples/nanogpt_diloco/sync_diloco_fsdp.py

## Streaming DiLoCo

Streaming DiLoCo is a more advanced version of DiLoCo which hides the communication time associated with the pseudo gradient reduction by
overlapping it with local training steps of the next step.
This technically applies slightly dirty parameter updates.
Given that each step is effectively an N=2 pipeline, we need to take special care of newcomers to properly insert the into the pipeline.
Specifically, given that the all reduce that completes at any given outer step `t` has commenced before the newcomer has joined, we need to ensure that the newcomer
gets to "eavesdrop" on the result, which only the pre-existing workers will see by performing a second shared state synchronization.

```python
# 1) Build your local model as usual:
local_model = GPT(config)
local_optimizer = AdamW(local_model, config)

# 2) Create separate "outer" parameters on CPU for aggregation:
outer_params = {}
for (name, local_param) in local_model.named_parameters():
    outer_params[name] = new_parameter_on_cpu(local_param.shape)

# 3) Create an "outer" optimizer, e.g. SGD:
outer_optimizer = SGD(outer_params, config)

# 4) Create PCCL communicator and connect:
communicator = Communicator(host=config["ccoip_host"], port=0)
communicator.connect()

# 5) Build a SharedState for the aggregator (outer params + momentum buffers), and set revision=0
shared_state_dict = collect_outer_params_and_buffers(outer_params, outer_optimizer)
tensor_info_entries = [TensorInfo.from_torch(tensor, name) for (name, tensor) in shared_state_dict.items()]
shared_state = SharedState(tensor_info_entries)
shared_state.revision = 0

# 6) Prepare structures for asynchronous all-reduce:
all_reduce_thread = None
last_pseudo_grads = None

# 7) The asynchronous training loop
for outer_iter in range(MAX_OUTER_ITERS):

    # (A) Possibly update topology if new peers are pending
    #     - If we detect pending peers, we must first join any running all-reduce thread
    #       so that update_topology won't conflict with an in-flight collective
    if communicator.are_peers_pending():
        if all_reduce_thread is not None:
            all_reduce_thread.join()  # Wait for the last reduce to finish
        communicator.update_topology()
        topology_updated = True
    else:
        topology_updated = False

    # If topology was updated, sync aggregator state so new peers can catch up
    if topology_updated:
        communicator.sync_shared_state(shared_state)
        copy_outer_params_into_local(local_model, outer_params)

    # (B) If there's a previous all-reduce in progress, wait for it to finish.
    #     Then apply the averaged “pseudo-grads” to outer_params and copy aggregator back locally.
    if all_reduce_thread is not None:
        all_reduce_thread.join()

        for name in outer_params:
            outer_params[name].grad = last_pseudo_grads[name]
        outer_optimizer.step()
        outer_optimizer.zero_grad()

        copy_outer_params_into_local(local_model, outer_params)

        # If new peers arrived mid-run, re-sync aggregator so they obtain the updated state
        if topology_updated and outer_iter > 0:
            communicator.sync_shared_state(shared_state)

    # (C) Perform local “inner” steps:
    for _ in range(config["inner_steps"]):
        local_loss = local_train_step(local_model, local_optimizer, batch_data())

    # (D) Compute new pseudo-grads = (outer_param - local_param).
    #     If no aggregator step was done yet (meaning can_outer_step == False in the code),
    #     and topology updated, then we must do a 'late_joiner' sync + reset local model to aggregator.
    can_outer_step = (all_reduce_thread is not None)
    if not can_outer_step and topology_updated and outer_iter > 0:
        #  - We are the late joiner, so we do another sync to get aggregator
        #  - Then we reset local parameters to that aggregator state
        communicator.sync_shared_state(shared_state)
        copy_outer_params_into_local(local_model, outer_params)
    
    pseudo_grad_dict = {}
    for name, local_param in local_model.named_parameters():
        diff = outer_params[name] - local_param.cpu()
        pseudo_grad_dict[name] = diff

    # (E) Launch a background thread to all-reduce these pseudo-grads
    def async_all_reduce(grad_dict):
        all_reduce_multiple_with_retry(communicator, list(grad_dict.values()), op=ReduceOp.AVG)

    last_pseudo_grads = pseudo_grad_dict
    all_reduce_thread = threading.Thread(
        target=async_all_reduce,
        args=(last_pseudo_grads,)
    )
    all_reduce_thread.start()

    # (F) Bump shared_state revision; possibly log, check for termination, etc.
    shared_state.revision += 1
    if termination_condition():
        break

communicator.close()
```

A full example implementing async DiLoCo is available here: https://github.com/PrimeIntellect-ai/pccl/blob/main/python/examples/nanogpt_diloco/async_diloco.py
