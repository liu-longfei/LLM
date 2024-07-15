CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 sft_qwen.py \
    --model_name_or_path /home/W0016778/code/dl/pretrain/qwen/qwen1.5-0.5B_output_wudao_pretrain_4-5/checkpoint-1935 \
    --data_path /model/data/datasets/jsonl/COIG/human_value_alignment_instructions_part1.jsonl \
    --bf16 True \
    --output_dir /home/W0016778/code/dl/sft/Qwen/output_qwen \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 10 \
    --learning_rate 3e-4 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "none" \
    --model_max_length 512 \
    --lazy_preprocess True \
    --gradient_checkpointing \
    # --deepspeed /home/W0016778/code/dl/sft/Qwen/ds_config_zero3.json
    # --use_lora ${USE_LORA} \
    # --q_lora ${Q_LORA} \
