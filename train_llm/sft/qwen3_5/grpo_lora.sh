SYSTEM_PROMPT="""You are a helpful math assistant. Solve the problem step by step and put your final answer within \\boxed{}."""
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
swift rlhf \
    --rlhf_type grpo \
    --model /home/user/ubuntu/pretrain/Qwen3.5-0.8B \
    --external_plugins /home/user/ubuntu/code/LLM/train_llm/sft/qwen3_5/gsm8k_plugin.py \
    --reward_funcs gsm8k_accuracy gsm8k_format \
    --columns '{"answer": "solution"}' \
    --enable_thinking false \
    --use_vllm false \
    --sleep_level 1 \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 16 \
    --torch_dtype bfloat16 \
    --dataset /home/user/ubuntu/datasets/gsm8k#1000 \
    --load_from_cache_file true \
    --max_length 2048 \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --save_steps 10 \
    --save_total_limit 100 \
    --logging_steps 1 \
    --warmup_ratio 0.0 \
    --dataloader_num_workers 1 \
    --num_generations 2 \
    --temperature 1.0 \
    --system "$SYSTEM_PROMPT" \
    --deepspeed zero3 \
    --log_completions true \
    --report_to tensorboard \
    --max_grad_norm 1.0 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --scale_rewards none \
    --do_eval false \
    --optim paged_adamw_8bit
