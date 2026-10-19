IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model /home/user/ubuntu/pretrain/Qwen3.5-0.8B \
    --enable_thinking false \
    --stream true
