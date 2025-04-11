# Rank 1
CUDA_VISIBLE_DEVICES="0,1" PYTHONPATH=../../.. PCCL_LOG_LEVEL=DEBUG BASE_PORT=10000 MASTER_PORT=10000 torchrun --nproc-per-node=2 --master_port=10000 sync_diloco_fsdp.py --config_path sync_config_gpu0.json &

# Rank 2
CUDA_VISIBLE_DEVICES="2,3" PYTHONPATH=../../.. PCCL_LOG_LEVEL=DEBUG BASE_PORT=20000 MASTER_PORT=20000 torchrun --nproc-per-node=2 --master_port=20000 sync_diloco_fsdp.py --config_path sync_config_other.json &


