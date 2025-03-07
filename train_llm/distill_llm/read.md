# GRPO

![5587be2bccbf495ba2e38e237e6ddbd5](file:///C:/Users/LW0016778/Pictures/Typedown/5587be2b-ccbf-495b-a2e3-8e237e6ddbd5.png?msec=1740968611095)

## 1、生成回答（ Generating completions）

        在每一步训练中，采样一批提示并针对每个提示生成一组 G 补全（表示为 oi​ ）。

## 2、计算优势（computing the advantage）

        对于每个 G 序列，我们使用奖励模型计算奖励。为了与奖励模型的比较性质相一致，通常在相同问题的输出之间的做比较训练。

![43d95cdcea3749b3bf83e4ae82825296](file:///C:/Users/LW0016778/Pictures/Typedown/43d95cdc-ea37-49b3-bf83-e4ae82825296.png?msec=1741141716353)

这种方法为该方法命名：组相对策略优化（GRPO：Group Relative Policy Optimization）。

## 3、估计KL散度（estimating the KL divergence）

KL散度表示表示两个分布的距离。训练模型（Policy）和参考模型（Ref policy）的KL散度就表示两个模型的参考性能。

![ca6132a04aa4483da863ee0ee5a18c93](file:///C:/Users/LW0016778/Pictures/Typedown/ca6132a0-4aa4-483d-a863-ee0ee5a18c93.png?msec=1741142494567)

> 源码：![7f4fd4b2730d4c74b718e5a697b3ff1a](file:///C:/Users/LW0016778/Pictures/Typedown/7f4fd4b2-730d-4c74-b718-e5a697b3ff1a.png?msec=1741146916581?msec=1741245396341?msec=1741246936804)

## 4、计算损失（computing the loss）

目标是**在确保模型接近参考模型的同时，最大化优势**。损失定义如下：

![a947a58cdccb4748ba1d880c74ba402f](file:///C:/Users/LW0016778/Pictures/Typedown/a947a58c-dccb-4748-ba1d-880c74ba402f.png?msec=1741142923640)

> 源码：
> 
> ![14d2b225d8564fce914f657d75ac74cc](file:///C:/Users/LW0016778/Pictures/Typedown/14d2b225-d856-4fce-914f-657d75ac74cc.png?msec=1741245377672?msec=1741246907264)

第一项表示缩放优势，第二项通过 KL 散度惩罚偏离参考策略的偏差。

在原始论文中，该公式被推广以考虑每一代之后的多次更新（表示为 μ ，可以在 GRPOConfig 中通过 `num_iterations` 设置）通过利用剪裁的代理目标函数：

![c34dd71e7fca461495309868e8508571](file:///C:/Users/LW0016778/Pictures/Typedown/c34dd71e-7fca-4614-9530-9868e8508571.png?msec=1741143228290)

clip(⋅,1−ϵ,1+ϵ) 确保更新不会过度偏离参考策略，通过限制策略裁减在 1−ϵ 和 1+ϵ 之间。当 μ=1 （在 TRL 中为默认值）时，裁剪的代理目标简化为原始目标。

丰富一些

未来展望

# 训练

## 训练简单任务

1、以Qwen2.5-0.5B-Instruct为例

设置输出奖励为 **答案是否正确**，**输出格式是否正确（软要求）**，**输出格式是否正确（硬要求）**

2、奖励函数

    # 答案是否正确
    def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        responses = [completion[0]['content'] for completion in completions]
        q = prompts[0][-1]['content']
        extracted_responses = [extract_xml_answer(r) for r in responses]
        return [1 if a in r else 0.0 for r, a in zip(extracted_responses, answer)]
    
    # 输出格式是否正确（软要求）
    def soft_format_reward_func(completions, **kwargs) -> list[float]:
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, r, re.DOTALL) for r in responses]
        return [2 if match else 0.0 for match in matches]
    
    # 输出格式是否正确（硬要求）
    def strict_format_reward_func(completions, **kwargs) -> list[float]:
        pattern = r"^\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, r, re.DOTALL) for r in responses]
        return [4 if match else 0.0 for match in matches]

3、训练日志

解读一下训练日志

![f57abc7ef143428ea12f928e54b2d302](file:///C:/Users/LW0016778/Pictures/Typedown/f57abc7e-f143-428e-a12f-928e54b2d302.png?msec=1741144393168)

## 配置器任务

1.使用torchrun全参会报错，lora可以训练，4张卡全部跑满

2.使用deepspeed stage2，可以训练全参，4张卡全部跑满

### Unsloth

> 链接：[Train your own R1 reasoning model locally (GRPO)](https://unsloth.ai/blog/r1-reasoning)
> 
> demo:[Google Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(3B)-GRPO.ipynb#scrollTo=DkIvEkIIkEyB)
> 
> 50G的显存就可以把代码跑起来

#### 训练记录

![04c85d86fca14f0096d7d7029aa7be49](file:///C:/Users/LW0016778/Pictures/Typedown/04c85d86-fca1-4f00-96d7-d7029aa7be49.png?msec=1741248365378)

#### 训练结果

1. 模型输出格式奖励在上升，模型学到了规范化格式
  
2. 模型的输出bom名称和真实名称奖励，还在震荡，可能原因
  
  * 奖励使用相似度可能不太合适
    
  * 训练的步数太少，没有等到模型的顿悟时刻
    




