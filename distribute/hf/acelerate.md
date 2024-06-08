# acelerate

Accelerate是一个库，只需添加四行代码，就可以在任何分布式配置中运行相同的PyTorch代码！简言之，大规模的训练和推理变得简单、高效和适应性强。

```python
+ from accelerate import Accelerator
+ accelerator = Accelerator()

+ model, optimizer, training_dataloader, scheduler = accelerator.prepare(
+     model, optimizer, training_dataloader, scheduler
+ )

  for batch in training_dataloader:
      optimizer.zero_grad()
      inputs, targets = batch
      inputs = inputs.to(device)
      targets = targets.to(device)
      outputs = model(inputs)
      loss = loss_function(outputs, targets)
+     accelerator.backward(loss)
      optimizer.step()
      scheduler.step()
```

Accelerate基于torch_xla和torch_distributed构建，可以承担繁重的工作，因此您不必编写任何自定义代码来适应这些平台。转换现有的代码库以利用DeepSpeed，执行完全分片的数据并行，并自动支持混合精度训练！

然后，可以通过Accelerate的CLI界面在任何系统上启动此代码：

```shell
accelerate launch {my_script.py}
```

## 配置Accelerate

安装后，您需要针对当前系统的培训设置方式配置Accelerate。要执行此操作，请运行以下操作并回答提示的问题：

```bash
accelerate config
```

要编写不包括DeepSpeed配置或在TPU上运行等选项的裸机配置，您可以快速运行：

```
python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"
```

Accelerate将自动利用可用的最大GPU数量，并设置混合精度模式。
要检查您的配置是否正常，请运行：

```
accelerate env
```

下面显示了一个示例输出，它描述了一台机器上的两个GPU，没有使用混合精度：

```bash
- `Accelerate` version: 0.11.0.dev0
- Platform: Linux-5.10.0-15-cloud-amd64-x86_64-with-debian-11.3
- Python version: 3.7.12
- Numpy version: 1.19.5
- PyTorch version (GPU?): 1.12.0+cu102 (True)
- `Accelerate` default config:
        - compute_environment: LOCAL_MACHINE
        - distributed_type: MULTI_GPU
        - mixed_precision: no
        - use_cpu: False
        - num_processes: 2
        - machine_rank: 0
        - num_machines: 1
        - main_process_ip: None
        - main_process_port: None
        - main_training_function: main
        - deepspeed_config: {}
        - fsdp_config: {}
```

## Quicktour

根据您的训练环境（torcrun、DeepSpeed等）和可用硬件，有多种方法可以启动和运行代码。Accelerate提供了一个统一的界面，用于在不同的分布式设置上启动和培训，使您能够专注于PyTorch培训代码，而不是复杂地将代码适应这些不同的设置。这允许您轻松地扩展PyTorch代码，以便使用GPU和TPU等硬件在分布式设置上进行训练和推理。Accelerate还提供了大模型推理，使加载和运行通常不适合内存的大型模型的推理更容易访问。

本快速教程介绍了Accelerate的三个主要功能：

- 用于分布式训练脚本的统一命令行启动界面
- 一个训练库，用于调整PyTorch训练代码以在不同的分布式设置上运行
- 大模型推理

## 统一启动界面

Accelerate通过Accelerate config命令生成的统一配置文件，自动为任何给定的分布式训练框架（DeepSpeed、FSDP等）选择适当的配置值。您也可以将配置值显式地传递到命令行，这在某些情况下很有帮助，比如使用SLURM。
但在大多数情况下，您应该始终首先运行accelerate配置，以帮助accelerate了解您的训练设置。

```bash
accelerate config
```

accelerate-config命令在Accelerates缓存文件夹中创建并保存一个default_config.yaml文件。此文件存储培训环境的配置，有助于Accelerate根据您的机器正确启动培训脚本。
配置好环境后，您可以使用加速测试来测试您的设置，该测试启动一个简短的脚本来测试分布式环境。

```bash
accelerate test
```

> 将--config_file添加到加速测试或加速启动命令中，以指定配置文件的位置（如果配置文件保存在缓存等非默认位置）。

一旦您的环境设置好，就可以通过accelerate launch来启动您的培训脚本！

```bash
accelerate launch path_to_script.py --args_for_the_script
```

要了解更多信息，请查看启动分布式代码教程，了解有关启动脚本的更多信息。

## 调整训练代码

Accelerate的下一个主要功能是Accelerator类，它可以调整PyTorch代码以在不同的分布式设置上运行。
您只需要在训练脚本中添加几行代码，即可使其在多个GPU或TPU上运行。

```python
+ from accelerate import Accelerator
+ accelerator = Accelerator()

+ device = accelerator.device
+ model, optimizer, training_dataloader, scheduler = accelerator.prepare(
+     model, optimizer, training_dataloader, scheduler
+ )

  for batch in training_dataloader:
      optimizer.zero_grad()
      inputs, targets = batch
-     inputs = inputs.to(device)
-     targets = targets.to(device)
      outputs = model(inputs)
      loss = loss_function(outputs, targets)
+     accelerator.backward(loss)
      optimizer.step()
      scheduler.step()
```

1. 在训练脚本开始时导入并实例化Accelerator类。Accelerator类初始化分布式训练所需的一切，并根据代码的启动方式自动检测您的训练环境（一台带有GPU的机器、一台带有多个GPU的机器，多台带有多个子GPU或TPU的机器等）。

```python
from accelerate import Accelerator

accelerator = Accelerator()
```

2. 删除对模型和输入数据的.cuda（）等调用。Accelerator类会自动将这些对象放置在适合您的设备上。

> 此步骤是可选的，但它被认为是允许Accelerate处理设备放置的最佳实践。您也可以在初始化加速器时通过传递device_placement=False来停用自动设备放置。如果要使用.to（device）在设备上显式放置对象，请确保使用accelerator.device。例如，如果在将模型放置在accelerator.device上之前创建了优化器，则TPU上的训练将失败。

> Accelerate的自动设备放置默认不使用非阻塞传输，这可能会导致潜在的不需要的CUDA同步。您可以通过在初始化Accelerator时传递设置为non_blocking=True的DataLoaderConfiguration来启用非阻塞传输。与往常一样，只有当数据加载器也设置了pin_memory=True时，非阻塞传输才会工作。请注意，如果使用从GPU到CPU的非阻塞传输导致在未准备好的张量上执行CPU操作，则可能会导致不正确的结果。

```python
device = accelerator.device
```

3. 一旦创建了所有相关的PyTorch对象（优化器、模型、数据加载器、学习率调度器），就将它们传递给prepare（）方法进行训练。此方法将模型封装在针对分布式设置优化的容器中，使用Accelerates版本的优化器和调度器，并创建数据加载器的碎片版本，用于在GPU或TPU之间分发。

```python
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)
```

4. Replace loss.backward() with backward() to use the correct backward() method for your training setup

```python
accelerator.backward(loss)
```

Read [Accelerate’s internal mechanisms](https://huggingface.co/docs/accelerate/concept_guides/internal_mechanism) guide to learn more details about how Accelerate adapts your code.

## 分布式评估

要执行分布式评估，请将验证数据加载器传递给prepare（）方法：

```python
validation_dataloader = accelerator.prepare(validation_dataloader)
```

分布式设置中的每个设备只接收一部分评估数据，这意味着您应该将预测与gather_for_metrics（）方法组合在一起。此方法要求每个进程上的所有张量都具有相同的大小，因此，如果您的张量在每个进程上具有不同的大小（例如，当动态填充到批中的最大长度时），则应使用pad_across_processes（）方法将张量填充到跨进程的最大大小。注意，张量需要是1D，并且我们将张量沿着第一维度连接起来。

```python
for inputs, targets in validation_dataloader:
    predictions = model(inputs)
    # Gather all predictions and targets
    all_predictions, all_targets = accelerator.gather_for_metrics((predictions, targets))
    # Example of use with a *Datasets.Metric*
    metric.add_batch(all_predictions, all_targets)
```

对于更复杂的情况（例如，2D张量，不想连接张量，3D张量的dict），可以在gather_for_metrics中传递use_gather_object=True。这将在收集后返回对象列表。请注意，将其与GPU张量一起使用并不受支持，而且效率低下。

> 数据集末尾的数据可能会重复，因此该批数据可以在所有工作人员中平均分配。gather_for_metrics（）方法会自动删除重复的数据，以计算更准确的度量。

## Big Model Inference

Accelerate的大模型推理有两个主要功能，init_empty_weights（）和load_checkpoint_and_dispatch（），用于加载通常不适合内存的大模型进行推理。

> Take a look at the [Handling big models for inference](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference) guide for a better understanding of how Big Model Inference works under the hood.

### Empty weights initialization

init_empty_weights（）上下文管理器通过创建模型骨架并在每次创建参数时将其移动和放置到PyTorch的元设备来初始化任何大小的模型。这样，就不会立即加载所有权重，一次只将模型的一小部分加载到内存中。
例如，加载一个空的Mixtral-8x7B模型所占用的内存比在CPU上完全加载模型和权重要少得多。

```python
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
```

### Load and dispatch weights

load_checkpoint_and_dispatch（）函数将完整或分片的检查点加载到空模型中，并在所有可用设备上自动分配权重。
device_map参数决定了每个模型层的放置位置，如果仍然没有足够的内存，则指定“auto”将它们作为内存映射张量先放在GPU上，然后放在CPU上，最后放在硬盘上。使用no_split_module _classes参数来指示哪些模块不应在设备之间进行拆分（通常是那些具有剩余连接的模块）。

```python
from accelerate import load_checkpoint_and_dispatch

model = load_checkpoint_and_dispatch(
    model, checkpoint="mistralai/Mixtral-8x7B-Instruct-v0.1", device_map="auto", no_split_module_classes=['Block']
)
```

## Next steps

现在您已经了解了Accelerate的主要功能，接下来的步骤可能包括：

- 查看教程，了解“加速”的轻松演练。如果你是分布式培训和图书馆的新手，这一点尤其有用。
- 深入阅读指南，了解如何针对特定用例使用Accelerate。
- 通过阅读概念指南，加深您对Accelerate内部工作方式的概念理解。
- 在API参考中查找类和命令，查看可用的参数和选项。

# Tutorials

## Overview

欢迎使用Accelerate教程！这些入门指南将帮助您了解使用Accelerate的最新进展。您将学习如何修改代码使其与API无缝配合，如何正确启动脚本，等等！

这些教程假定了Python的一些基本知识以及对PyTorch框架的熟悉程度。
如果您对Accelerate有任何疑问，请随时加入我们的论坛并向社区提问。

## Add Accelerate to your code

每个分布式培训框架都有自己的工作方式，这可能需要编写大量自定义代码来适应PyTorch培训代码和培训环境。Accelerate提供了一种友好的方式来与这些分布式培训框架对接，而无需了解每个框架的具体细节。Accelerate会为您处理这些细节，因此您可以专注于培训代码，并将其扩展到任何分布式培训环境。
在本教程中，您将学习如何使用Accelerate调整现有的PyTorch代码，并轻松地进行分布式系统培训！您将从一个基本的PyTorch训练循环开始（它假设所有的训练对象（如模型和优化器）都已经设置好了），并逐步将Accelerate集成到其中。

```python
device = "cuda"
model.to(device)

for batch in training_dataloader:
    optimizer.zero_grad()
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

## Accelerator

Accelerator是用于调整代码以使用Accelerate的主要类。它知道您正在使用的分布式设置，例如不同进程的数量和您的硬件类型。该类还提供了对许多必要方法的访问，这些方法使PyTorch代码能够在任何分布式培训环境中工作，以及跨设备管理和执行流程。
这就是为什么您应该始终从在脚本中导入和创建加速器实例开始。

```python
from accelerate import Accelerator

accelerator = Accelerator()
```

Accelerator还知道将PyTorch对象移动到哪个设备，因此建议让Accelerate为您处理此问题。

```python
- device = "cuda"
+ device = accelerator.device
  model.to(device)
```

## Prepare PyTorch objects

接下来，您需要为分布式训练准备PyTorch对象（模型、优化器、调度器等）。prepare（）方法负责将模型放置在用于训练设置的适当容器（如单GPU或多GPU）中，调整优化器和调度器以使用Accelerate的AcceleratedOptimizer和AcceleratedScheduler，并创建一个可以跨进程分片的新数据加载器。

> Accelerate仅准备从各自PyTorch类（如torch.optim）继承的对象。优化器。

PyTorch对象的返回顺序与它们的发送顺序相同。

```python
model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)
```

## Training loop

最后，在训练循环中删除对输入和目标的to(device)调用，因为Accelerate的DataLoader类会自动将它们放置在正确的设备上。您还应该用Accelerate的backward（）方法来代替通常的backwards（）方法，该方法会根据您的分布式设置（例如，DeepSpeed或Megatron）来缩放梯度并使用适当的backwarding（）方法。

```python
inputs = inputs.to(device)
-   targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
-   loss.backward()
+   accelerator.backward(loss)
```

把所有东西放在一起，你的新加速训练循环现在应该是这样的！

```python
from accelerate import Accelerator
accelerator = Accelerator()

device = accelerator.device
model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)

for batch in training_dataloader:
    optimizer.zero_grad()
    inputs, targets = batch
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
```

## Training features

Accelerate提供了其他功能，如梯度积累、梯度剪裁、混合精度训练等，您可以将其添加到脚本中，以提高训练效果。让我们探究一下这三个特性。

### Gradient accumulation

梯度累积使您能够通过在更新权重之前累积多个批次的梯度来训练较大的批次大小。这对于克服内存限制非常有用。要在Accelerate中启用此功能，请在Accelerator类中指定gradient_accumulation_steps参数，并将accumulate（）上下文管理器添加到脚本中。

```python
+ accelerator = Accelerator(gradient_accumulation_steps=2)
  model, optimizer, training_dataloader = accelerator.prepare(model, optimizer, training_dataloader)

  for input, label in training_dataloader:
+     with accelerator.accumulate(model):
          predictions = model(input)
          loss = loss_function(predictions, label)
          accelerator.backward(loss)
          optimizer.step()
          scheduler.step()
          optimizer.zero_grad()
```

### Gradient clipping

梯度剪裁是一种防止“梯度爆炸”的技术，Accelerate提供：

- clipgrad_value（）将梯度剪裁到最小值和最大值
- clipgrad_norm（），用于将梯度归一化为某个值

### Mixed precision

混合精度通过使用像fp16（半精度）这样的低精度数据类型来计算梯度来加速训练。为了获得Accelerate的最佳性能，应该在模型内部计算损失（就像在transformers模型中一样），因为模型外部的计算是完全精确的。
设置要在Accelerator中使用的混合精度类型，然后使用autocast（）上下文管理器将值自动转换为指定的数据类型。

Accelerate启用自动混合精度，因此只有在除了已经处理缩放的backward（）在丢失时执行的操作之外，还有其他混合精度操作时，才需要autocast（）。

```python
+ accelerator = Accelerator(mixed_precision="fp16")
+ with accelerator.autocast():
      loss = complex_loss_function(outputs, target):
```

## Save and load

Accelerate还可以在训练完成后保存和加载模型，也可以保存模型和优化器状态，这对恢复训练很有用。

### Model

所有过程完成后，在保存模型之前，请使用unwrap_model（）方法打开模型，因为prepare（）方法将模型封装到用于分布式训练的适当接口中。如果不打开模型，保存模型状态字典也会保存较大模型中任何潜在的额外层，并且无法将权重加载回基本模型中。

您应该使用save_model（）方法来打开和保存模型状态字典。此方法还可以将模型保存为分片检查点或安全张量格式。

#### single checkpoint

```python
accelerator.wait_for_everyone()
accelerator.save_model(model, save_directory)
```

对于Transformers库中的模型，请使用save_pretrained方法保存该模型，以便可以使用from_pretrained方法重新加载该模型。

```python
from transformers import AutoModel

unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(
    "path/to/my_model_directory",
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
)

model = AutoModel.from_pretrained("path/to/my_model_directory")
```

要加载权重，请在加载权重之前先使用unwrap_model（）方法打开模型。所有模型参数都是张量的参考，所以这会在模型中加载权重。

```python
unwrapped_model = accelerator.unwrap_model(model)
path_to_checkpoint = os.path.join(save_directory,"pytorch_model.bin")
unwrapped_model.load_state_dict(torch.load(path_to_checkpoint))
```

#### single checkpoint

将safe_serialization设置为True，以安全张量格式保存模型。

```python
accelerator.wait_for_everyone()
accelerator.save_model(model, save_directory, max_shard_size="1GB", safe_serialization=True)
```

要加载分片检查点或安全张量格式的检查点，请使用load_checkpoint_in_model（）方法。此方法允许您将检查点加载到特定设备上。

```python
load_checkpoint_in_model(unwrapped_model, save_directory, device_map={"":device})
```

### State

在训练过程中，您可能希望保存模型、优化器、随机生成器和潜在的学习率调度器的当前状态，以便在同一脚本中恢复它们。您应该将save_state（）和load_state（）方法添加到脚本中，以保存和加载状态。
要进一步自定义通过save_state（）保存状态的位置和方式，请使用ProjectConfiguration类。例如，如果启用了automatic_checkpoint_naming，则每个保存的检查点都存储在Accelerator.project_dir/checkpoint/checkpoint_{checkpoint_number}中。
要存储的任何其他有状态项都应该使用register_for_checkpointing（）方法注册，以便保存和加载它们。传递给此方法以进行存储的每个对象都必须具有load_state_dict和state_dict函数。


# Launching your 🤗 Accelerate scripts

在上一个教程中，向您介绍了如何修改当前的训练脚本以使用🤗 加快该代码的最终版本如下所示：

```python
from accelerate import Accelerator

accelerator = Accelerator()

model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)

for batch in training_dataloader:
    optimizer.zero_grad()
    inputs, targets = batch
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
```

但是，你如何运行这个代码，并让它利用可用的特殊硬件？
首先，您应该将上面的代码重写为一个函数，并使其可以作为脚本调用。例如

```python
 from accelerate import Accelerator
  
+ def main():
      accelerator = Accelerator()

      model, optimizer, training_dataloader, scheduler = accelerator.prepare(
          model, optimizer, training_dataloader, scheduler
      )

      for batch in training_dataloader:
          optimizer.zero_grad()
          inputs, targets = batch
          outputs = model(inputs)
          loss = loss_function(outputs, targets)
          accelerator.backward(loss)
          optimizer.step()
          scheduler.step()

+ if __name__ == "__main__":
+     main()
```

接下来，您需要通过加速启动来启动它。

> 建议您在使用加速启动之前先运行加速配置，以根据自己的喜好配置环境。否则，Accelerate将根据您的系统设置使用非常基本的默认值。

## Using accelerate launch

Accelerate有一个特殊的CLI命令，可帮助您通过加速启动在系统中启动代码。这个命令包含了在各种平台上启动脚本所需的所有不同命令，而不必记住每个命令是什么。

> 如果你自己熟悉在PyTorch中启动脚本，比如torchrun，你仍然可以这样做。不需要使用加速发射。

您可以使用以下命令快速启动脚本：

```bash
accelerate launch {script_name.py} --arg1 --arg2 ...
```

只需在命令开始时设置加速启动，然后像往常一样将其他参数和参数传递到脚本中！
由于这会运行各种torch生成方法，因此所有预期的环境变量也可以在这里进行修改。例如，以下是如何使用单个GPU加速启动：

```bash
CUDA_VISIBLE_DEVICES="0" accelerate launch {script_name.py} --arg1 --arg2 ...
```

您也可以在不首先执行加速配置的情况下使用加速启动，但可能需要手动输入正确的配置参数。在这种情况下，Accelerate将为您做出一些超参数决策，例如，如果GPU可用，它将默认使用所有GPU，而不会出现混合精度。以下是如何使用所有GPU并在禁用混合精度的情况下进行训练：

```bash
accelerate launch --multi_gpu {script_name.py} {--arg1} {--arg2} ...
```

或者通过指定要使用的GPU数量：

```bash
accelerate launch --num_processes=2 {script_name.py} {--arg1} {--arg2} ...
```

为了更具体，您应该自己传递所需的参数。例如，以下是如何使用混合精度在两个GPU上启动相同的脚本，同时避免所有警告：

```bash
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 {script_name.py} {--arg1} {--arg2} ...
```

有关可以传入的参数的完整列表，请运行：

```bash
accelerate launch -h
```

> 即使您的代码中没有使用Accelerate，您仍然可以使用启动器来启动脚本！

为了可视化这种差异，早期在多gpu上的accelerate launch与torcrun类似

```bash
MIXED_PRECISION="fp16" torchrun --nproc_per_node=2 --num_machines=1 {script_name.py} {--arg1} {--arg2} ...
```

您还可以将启动CLI作为python模块本身来启动脚本，从而能够传入其他特定于python的启动行为。要执行此操作，请使用accelerate.commands.slaunch而不是accelerate launch：

```bash
python -m accelerate.commands.launch --num_processes=2 {script_name.py} {--arg1} {--arg2}
```

如果您想使用任何其他python标志执行脚本，也可以像-m一样传入它们，例如下面的启用无缓冲stdout和stderr的示例：

```bash
python -u -m accelerate.commands.launch --num_processes=2 {script_name.py} {--arg1} {--arg2}
```

> 你也可以在CPU上运行你的代码！这有助于在玩具模型和数据集上进行调试和测试。

```bash
accelerate launch --cpu {script_name.py} {--arg1} {--arg2}
```

## Why you should always use accelerate config

为什么它对您应该始终运行accelerate config有用？
还记得早些时候的呼吁吗？配置后，要使用所需的部分运行该脚本，您只需要直接使用accelerate launch，而无需传递任何其他信息：

```bash
accelerate launch {script_name.py} {--arg1} {--arg2} ...
```

## Custom Configurations

如前所述，加速启动应该主要通过将设置配置与加速配置命令相结合来使用。这些配置保存到Accelerate缓存文件夹中的default_config.yaml文件中。此缓存文件夹位于（按优先级降序排列）：

- 环境变量HF_HOME的内容后缀为accelerate。
- 如果它不存在，则环境变量XDG_CACHE_HOME的内容后缀为huggingface/accept。
- 如果这也不存在，文件夹~/.cache/huggingface/accept。

要具有多个配置，可以将标志--config_file与自定义yaml的位置一起传递给加速启动命令。

对于使用fp16实现混合精度的单机上的两个GPU，示例yaml可能如下所示：

```bash
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: MULTI_GPU
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```

从该自定义yaml文件的位置启动脚本如下所示：

```bash
accelerate launch --config_file {path/to/config/my_config_file.yaml} {script_name.py} {--arg1} {--arg2} ...
```

## Multi-node training

Accelerate的多节点训练类似于torchrun的多节点培训。启动多节点训练运行的最简单方法是执行以下操作：

- 将代码库和数据复制到所有节点。（或将它们放在共享文件系统上）
- 在所有节点上设置您的python包。
- 首先在主单节点上运行加速配置。指定节点数后，系统将要求您指定每个节点的级别（主/主节点为0），以及主进程的IP地址和端口。这是工作节点与主进程通信所必需的。之后，您可以在所有节点上复制或发送此配置文件，将machine_rank更改为1、2、3等，以避免必须运行命令（或者直接按照他们的指示使用torchrun启动）

完成后，您可以通过在所有节点上运行accelerate launch（或torcrun）来开始多节点训练。

> 命令必须在所有节点上运行才能启动，而不仅仅是从主节点运行。您可以使用类似SLURM或其他流程执行器的东西来包装此需求，并从一个命令中调用所有内容。

> 建议使用主节点的intranet IP而不是公共IP，以获得更好的延迟。这是在主节点上运行hostname-I时看到的192.168.x.x或172.x.x.x地址。

To get a better idea about multi-node training, check out our example for [multi-node training with FSDP](https://huggingface.co/blog/ram-efficient-pytorch-fsdp).
