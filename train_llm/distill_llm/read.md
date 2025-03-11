# GRPO

        Group Relative Policy Optimization（群体相对策略优化）是一种专门用于增强大型语言模型（LLMs）推理能力的强化学习算法。通过评估一组回答之间的相对表现来优化模型。这种方法使训练更加高效，使GRPO特别适合需要复杂问题解决和长链思维推理的任务。

## GRPO流程

1、生成回答（ Generating completions）

        在每一步训练中，采样一批提示并针对每个提示生成一组 G潜在回答（表示为 oi​ ）。

2、计算优势（computing the advantage）

        对于每个 G 序列，我们使用奖励函数计算奖励。为了与奖励模型的比较性质相一致，通常在相同问题的输出之间的做比较训练，是回答在组中的相对表现，而不是孤立地评估单个回答。

![43d95cdcea3749b3bf83e4ae82825296](file:///C:/Users/LW0016778/Pictures/Typedown/43d95cdc-ea37-49b3-bf83-e4ae82825296.png?msec=1741690189576)

> 源码：
> 
>     # Apply weights to each reward function's output and sum
>     rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
>     
>     # Compute grouped-wise rewards
>     mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
>     std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
>     
>     # compute the advantages
>     advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

这种方法为该方法命名：组相对策略优化（GRPO：Group Relative Policy Optimization）。消除了对独立奖励模型的需求，使GRPO既高效又稳健。通过在组内引入竞争，GRPO推动模型不断提升其推理能力。

3、估计KL散度（estimating the KL divergence）

KL散度表示表示两个分布的距离。训练模型（Policy）和参考模型（Ref policy）的KL散度就表示两个模型的参考性能。

![ca6132a04aa4483da863ee0ee5a18c93](file:///C:/Users/LW0016778/Pictures/Typedown/ca6132a0-4aa4-483d-a863-ee0ee5a18c93.png?msec=1741690189576)

> 源码：![7f4fd4b2730d4c74b718e5a697b3ff1a](file:///C:/Users/LW0016778/Pictures/Typedown/7f4fd4b2-730d-4c74-b718-e5a697b3ff1a.png?msec=1741146916581?msec=1741245396341?msec=1741690189577)
> 
>     per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
> 
> * $\frac{x}{y} $ = $e^{log\frac{x}{y}}$ = $e^{logx - logy}$
>   
> * $log\frac{x}{y}$ = $logx$ - $logy$
>   
> 
> **在大模型中每个token的预测概率为什么要取对数：**
> 
> * **数值稳定性**：在计算多个概率的乘积时，由于概率值通常都在 0 到 1 之间，多个小数相乘会导致结果迅速变小，可能会出现下溢问题，即结果小于计算机能够表示的最小数值。而取对数后，乘法就转化为加法，避免了多个小数值直接相乘带来的数值稳定性问题。
>   
> * **优化计算效率**：在进行梯度计算等优化操作时，对数函数的导数形式相对简单，计算更加高效。
>   

4、计算损失（computing the loss）

目标是**在确保模型接近参考模型的同时，最大化优势**。损失定义如下：

![a947a58cdccb4748ba1d880c74ba402f](file:///C:/Users/LW0016778/Pictures/Typedown/a947a58c-dccb-4748-ba1d-880c74ba402f.png?msec=1741690189578)

第一项表示缩放优势，第二项通过 KL 散度惩罚偏离参考策略的偏差。

* 第一项：根据计算得到的相对优势 ，增加具有正相对优势的动作的概率，同时减少具有负相对优势的动作的概率。
  
* 第二项：防止模型更新过于剧烈，引入了KL散度约束。通过限制模型更新前后之间的KL散度，确保变化在可控范围内。
  

> 源码：
> 
>     per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
>     per_token_loss = -(per_token_loss - self.beta * per_token_kl)
>     loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
> 
> ![14d2b225d8564fce914f657d75ac74cc](file:///C:/Users/LW0016778/Pictures/Typedown/14d2b225-d856-4fce-914f-657d75ac74cc.png?msec=1741245377672?msec=1741690189578)

在**DeepSeekMath**原始论文中，该公式被推广以考虑每一代之后的多次更新（表示为 μ ，可以在 GRPOConfig 中通过 `num_iterations` 设置）通过利用剪裁的代理目标函数：

![c34dd71e7fca461495309868e8508571](file:///C:/Users/LW0016778/Pictures/Typedown/c34dd71e-7fca-4614-9530-9868e8508571.png?msec=1741690189579)

clip(⋅,1−ϵ,1+ϵ) 确保更新不会过度偏离参考策略，通过限制策略裁减在 1−ϵ 和 1+ϵ 之间。当 μ=1 （在 TRL 中为默认值）时，裁剪的代理目标简化为原始目标。

## GRPO图示

![5587be2bccbf495ba2e38e237e6ddbd5](file:///C:/Users/LW0016778/Pictures/Typedown/5587be2b-ccbf-495b-a2e3-8e237e6ddbd5.png?msec=1740968611095?msec=1741690189579)

* * *

**GRPO与PPO对比**

![bc1c0ea5c7052810b5f2ea6482be5b75](https://i-blog.csdnimg.cn/img_convert/bc1c0ea5c7052810b5f2ea6482be5b75.png)

**GRPO 与PPO 的主要区别有：**

* GRPO 省略了 value function model.
  
* GRPO reward 计算，改成了一个q 生成多个r, 然后reward 打分。
  
* PPO 优势函数计算时，KL 是包含在GAE内部的。 GRPO 直接挪到了外面，同时修改了计算方法。
  

传统的强化学习算法（如Proximal Policy Optimization，PPO）在应用于LLMs的推理任务时面临着重大挑战：

1. **依赖奖励模型**
  
  * PPO需要一个独立的奖励模型来评估每个回答的价值，这使内存和计算需求增加了一倍。
    
  * 训练奖励模型非常复杂且容易出错，尤其是在需要对主观或细微差别进行评价的任务中。
    
2. **高昂的计算成本**
  
  * 强化学习流程通常需要大量计算资源来迭代评估和优化回答。
    
  * 将这些方法扩展到更大的LLMs会进一步加剧成本。
    
3. **可扩展性问题**
  
  * 绝对奖励评估难以应对多样化任务，使得跨推理领域的泛化变得困难。

**GRPO如何应对这些挑战：**

* **无批评者优化：** GRPO通过比较组内回答，消除了对批评者模型的需求，显著降低了计算开销。
  
* **相对评估：** GRPO不依赖外部评价者，而是利用组内动态来评估每个回答在同一批次中的相对表现。
  
* **高效训练：** 通过专注于组内优势，GRPO简化了奖励估计流程，使其对大型模型的训练更快且更具可扩展性。
  

## 代码Demo

1、数据集

openai/gsm8k：GSM8K 由 **8.5K** 高质量的小学数学问题组成，这些问题都是由人类写手创造的。我们将这些问题分为 **7.5K 训练问题和 1K 测试问题**。

> 数据展示：
> 
> 问题：Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
> 
> 问题翻译：娜塔莉亚在四月份向她的 48 个朋友出售了发夹，然后她在五月份出售的发夹数量是四月份的一半。娜塔莉亚在四月份和五月份一共出售了多少个发夹？
> 
> 答案：Natalia sold 48/2 = <<48/2=24>>24 clips in May.Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.#### 72
> 
> 答案翻译：娜塔莉亚在五月份出售了 48÷2 = 24 个发夹。娜塔莉亚在四月份和五月份一共出售了 48 + 24 = 72 个发夹。#### 72

2、以Qwen2.5-0.5B-Instruct为例

设置输出奖励为 **答案是否正确**，**输出格式是否正确（软要求）**，**输出格式是否正确（硬要求）**

3、奖励函数

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

4、训练日志

解读一下训练日志

![f57abc7ef143428ea12f928e54b2d302](file:///C:/Users/LW0016778/Pictures/Typedown/f57abc7e-f143-428e-a12f-928e54b2d302.png?msec=1741144393168?msec=1741690189579)

* reward曲线：300步之前都在来回震荡，300之后有一个“顿悟”时刻
  
* soft_format和strict_format和总的reward一样，会顿悟
  
* 回答正确的奖励函数，相对比较平滑
  
* 训练的loss会震荡上升，这个和huggingface训练的曲线是一致的
  

## 配置器GRPO训练

1、数据

配件信息、配件原则和用户需求 生成 配置推荐

2、模型

使用的是使用R1蒸馏的数据微调的Qwen2.5-7B-Instruct模型

3、奖励函数

    # 设置格式奖励函数
    def soft_format_reward_func(completions, **kwargs) -> list[float]:
        pattern = r"```json(.*)```"
        responses = [compltion[0]["content"] for completion in completions]
        matches = [re.search(pattern, r, re.DOTALL) for r in responses]
        format_reward_list = [0.2 if match else 0.0 for match in matches]
        logger.info(f"soft_format生成奖励: {format_reward_list}")
        return format_reward_list
    
    # 设置格式奖励函数
    def strict_format_reward_func(completions, **kwargs) -> list[float]:
        # 设置json格式中的数据类型是json，不是列表。（部分代码）
        if isinstance(json_data, dict):
            strict_reward_list.append(0.5)
        else:
            strict_reward_list.append(0)
    
    # 设置bom和name一致的奖励函数
    def bom_name_reward_func(completions, **kwargs):
        # cosine_similarity是模型生成的bom名称和数据库中名称的余弦相似度 （部分代码）
        if cosine_similarity < 0.8:
            sub_reward_list.append(0.0)
        elif cosine_similarity < 0.9:
            sub_reward_list.append(0.6)
        else:
            sub_reward_list.append(1.0)

4、训练日志

解读一下训练日志

![04d7a528a87a42efaa75b416ce79faf4](file:///C:/Users/LW0016778/Pictures/Typedown/04d7a528-a87a-42ef-aa75-b416ce79faf4.png?msec=1741690189580)

* 同上训练的loss会上升
* 模型输出的bom-name奖励函数曲线来看，模型在更新，原有的能力会下降，随着参数更新的更新模型会重新学习到这种特征
* 模型更新前期，会先学习比较容易的特征
* 从三个奖励函数的曲线来看，给奖励值高的在参数更新后期占主导地位
* temperature越大，模型发散性越好，越有利于模型的反思（有资料显示）
* 使用GRPO训练后的模型，在配置器数据上指标有些上升

**配置器训练的问题：**

* 训练比较吃资源，qwen2.5-7B蒸馏后的模型4卡80G跑不起来。使用deepspeed、LoRA、Unsloth才能训起来
  
* 没有直接的奖励函数
  

下一步：

* unsloth不支持多卡，训练效果需要好几天，后期可以QLora和deepspeed结合加速训练
  
* 直接的奖励函数帮助模型更好的收敛
