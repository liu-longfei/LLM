# ✅ SmolVLA 优质学习资源（按对你的价值排序，优先代码/链路/离线、SO101 实操）
> 你的场景：LeRobot + SO101/SO100、本地模型离线加载、lerobot-train / 推理脚本、rename_map、多相机、flow matching，我优先挑**能直接看懂源码、能复现、踩坑对齐你现在问题**的资源

## 🔥 【国内最强源码逐行解析｜首选，最贴合你现在的问题】
1. **Laumy 博客：LeRobot SmolVLA:从训练到推理链路剖析**（中文，唯一逐行拆解 `modeling_smolvla.py`）
链接：[https://www.laumy.tech/2743.html/lerobot-smolvla%E7%AD%96%E7%95%A5/](https://www.laumy.tech/2743.html/lerobot-smolvla%E7%AD%96%E7%95%A5/)
亮点：
- 拆解 `embed_prefix / embed_suffix / forward / flow matching` 完整前向
- 讲清楚 VLM backbone（SmolVLM2）和 Action Expert 怎么拼接、KV cache 怎么复用
- 正好对应你 `SmolVLAPolicy.from_pretrained` 离线加载报错、模型内部硬编码 VLM 地址这个坑
- 有代码片段、变量含义、输入输出维度，**学源码第一优先**

## 📄 中文文档（命令、参数、SO101 微调，适合写脚本）
2. **dctx-team LeRobot 中文文档 SmolVLA 章节**
链接：[https://dctx-team.github.io/lerobot-zh/zh/smolvla/](https://dctx-team.github.io/lerobot-zh/zh/smolvla/)
亮点：
- 原生 `lerobot-train` 命令、参数说明、batch、steps、wandb 开关
- 数据集、repo_id 用法，和你 ttt.sh 脚本几乎一模一样
- 缺点：偏使用文档，**不深入源码**，适合先跑通命令

3. **Kuavo 文档 SmolVLA**（架构+数据流图）
链接：[https://www.kdc.docs.lejurobot.com/policy/smolvla/](https://www.kdc.docs.lejurobot.com/policy/smolvla/)
亮点：画了完整数据流：图像/SigLIP → prefix；动作噪声+timestep → suffix；flow matching 去噪输出 action chunk，适合快速建立架构认知

## 🌍 国外高质量实操教程（SO101 / SO100 真实机械臂，非常适合你）
4. **Arm 官方 Learning Path：Finetune SmolVLA with SO-101**（强烈推荐，官方工程向）
链接：[https://learn.arm.com/learning-paths/laptops-and-desktops/finetune-smolvla-lerobot/](https://learn.arm.com/learning-paths/laptops-and-desktops/finetune-smolvla-lerobot/)
亮点：
- 从头到尾：录制演示数据集 → rename 相机 → 微调 → rollout 真实臂
- 专门讲：VLM 冻结、Action Expert 训练、异步推理、action chunk 机制
- 正好匹配你的硬件：SO101 双臂、top/wrist 双相机
- 英文，但是步骤清晰，是工业界落地标准流程

5. **Seeed Studio Wiki：SmolVLA on SO-ARM101**（矽递官方，硬件+软件一体）
链接：[https://wiki.seeedstudio.com/robotics/paper_reading_smolvla/](https://wiki.seeedstudio.com/robotics/paper_reading_smolvla/)
亮点：讲异步推理、RobotClient / PolicyServer 分离，就是你后面 rollout 要用到的架构；区分单步动作和 chunk 动作

6. **VnRobo Blog：SmolVLA train on consumer GPU**（24G 消费卡微调，显存优化）
链接：[https://vnrobo.com/en/blog/vla-lerobot-12-smolvla-training](https://vnrobo.com/en/blog/vla-lerobot-12-smolvla-training)
亮点：12G/24G 卡显存配置、lr、save_freq、batch 调参，适合你本地单卡训练

## 📜 原始论文 + 官方源码（底层原理）
7. **SmolVLA arXiv 论文**：[https://arxiv.org/pdf/2506.01844.pdf](https://arxiv.org/pdf/2506.01844.pdf)
核心：异步推理、flow matching action expert、数据构成、LIBERO benchmark
8. **HuggingFace LeRobot 官方仓库（smolvla 源码本体）**
[https://github.com/huggingface/lerobot/tree/main/lerobot/policies/smolvla](https://github.com/huggingface/lerobot/tree/main/lerobot/policies/smolvla)
> `modeling_smolvla.py` 就是你 infer.py 加载的文件，所有博客都是基于这个文件解析

## 🎬 视频（国内B站/抖音，适合入门，**深度源码讲解很少**）
- 抖音【小工蚁】SmolVLA 入门介绍：偏概念科普，不读代码，适合先建立认知，**不能解决你离线加载报错**
- B站关键词搜索：`SmolVLA LeRobot SO100`，大部分是跑通演示视频，逐行源码解析的视频极少，**视频不推荐作为主力**

# 📌 给你的学习顺序建议（贴合你当前排错流程）
1. 先看 Laumy 博客（看懂 `embed_prefix` 里面是怎么加载内置 SmolVLM2，理解为什么离线会联网拉取 HuggingFaceTB/SmolVLM2-500M-Video-Instruct）
2. 再看 Arm 官方 SO101 教程，对齐数据集 rename、相机映射
3. 对照 dctx 中文文档改你的 `lerobot-train` 脚本
4. 最后看论文理解 flow matching、异步推理

# 🧩 额外补充（你现在最痛的离线问题，上面博客里没有直接写）
SmolVLAPolicy 内部 config 写死了 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`，**不从你传的 model_id 自动覆盖 VLM 路径**，有两个方案：
- 方案A：修改 `smolvla/config.json`，把里面的 vlm_name 改成你本地绝对路径
- 方案B：在 `from_pretrained` 时传入 `config_overrides` 覆盖 vlm 路径，并且 `local_files_only=True`

如果你想要，我可以直接给你**可运行的 infer.py 离线版本（不需要联网拉 SmolVLM2）**。