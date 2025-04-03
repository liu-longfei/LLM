torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    --nnodes=1 \
    --node_rank=0 \
    qwen2.5_grpo_train2_lora.py
