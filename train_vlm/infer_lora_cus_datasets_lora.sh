# 指定数据集推理
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift infer \
    --model /home/user/ubuntu/pretrain/InternVL3_5-1B \
    --adapters /home/user/ubuntu/pretrain/train_model/invernvl3_5_output/v7-20251016-222140/checkpoint-990 \
    --stream true \
    --infer_backend lmdeploy \
    --val_dataset /home/user/ubuntu/code/LLM/train_vlm/cus_internvl3_5.jsonl#10 \
    --max_new_tokens 2048 \
    --result_path /home/user/ubuntu/code/LLM/train_vlm/eval_datasets/res_2.jsonl
