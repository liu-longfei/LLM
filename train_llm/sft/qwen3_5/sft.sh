# 76GiB
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model /home/user/ubuntu/pretrain/Qwen3.5-0.8B \
    --tuner_type full \
    --dataset '/home/user/ubuntu/datasets/alpaca-gpt4-data-zh#5000' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 1024 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2 \
    --model_author swift \
    --model_name swift-robot \
    --do_eval false \
    --optim paged_adamw_8bit

