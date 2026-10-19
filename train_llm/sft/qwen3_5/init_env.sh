pip install ms-swift
pip install "transformers>=5.9" "qwen_vl_utils>=0.0.14" peft liger-kernel

# flash-linear-attention
# 若出现训练缓慢的问题请参考：https://github.com/fla-org/flash-linear-attention/issues/758
# 请使用python3.12: https://github.com/fla-org/flash-linear-attention/issues/121
pip install "flash-linear-attention>=0.4.2" --no-build-isolation
# 昇腾 NPU GDN 请安装 main 分支最新版本（替换上一条命令）
# pip install -U git+https://github.com/fla-org/flash-linear-attention.git --no-build-isolation

# causal_conv1d
# pip install -U git+https://github.com/Dao-AILab/causal-conv1d --no-build-isolation

# flash-attention
pip install "flash-attn==2.8.3" --no-build-isolation

# deepspeed训练
pip install deepspeed

# vllm (torch2.10) for inference/deployment/RL
pip install "vllm>=0.17.0"

# -i https://pypi.tuna.tsinghua.edu.cn/simple