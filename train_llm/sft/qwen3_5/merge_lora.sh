# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
swift export \
    --adapters /home/user/ubuntu/code/LLM/train_llm/sft/qwen3_5/output/Qwen3.5-0.8B-swift/v2-20260831-225934/checkpoint-350 \
    --merge_lora true