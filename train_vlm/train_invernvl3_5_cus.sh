# 33G
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model /home/user/ubuntu/pretrain/InternVL3_5-1B \
    --dataset '/home/user/ubuntu/code/LLM/train_vlm/cus_internvl3_5.jsonl' \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --torch_dtype float16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 2 \
    --lora_alpha 4 \
    --target_modules 'down_proj' \
    --freeze_vit true \
    --gradient_accumulation_steps 1 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir /home/user/ubuntu/pretrain/train_model/invernvl3_5_output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2