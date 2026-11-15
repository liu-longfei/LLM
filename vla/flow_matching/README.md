# CIFAR-10 Flow Matching Training and Inference

这个项目使用条件 U-Net 和 Flow Matching 方法，在 CIFAR-10 数据集上训练一个可以生成彩色图像的模型。

- 数据集：CIFAR-10 RGB 彩色图
- 图像尺寸：`32 x 32`
- 输入通道：`3`
- 类别数量：`10`
- 生成方式：从噪声开始，使用 Euler 积分逐步采样

---

## 1. 环境依赖

需要安装以下依赖：

```bash
pip install torch torchvision
```

如果需要查看生成的图片，可以直接用常见图片查看器打开 PNG 文件。

---

## 2. 数据集说明

默认使用的 CIFAR-10 数据集路径是：

```text
/home/user/ubuntu/datasets/cifar-10-python
```

目录结构应为：

```text
/home/user/ubuntu/datasets/cifar-10-python
└── cifar-10-batches-py
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    ├── test_batch
    └── batches.meta
```

当前数据集已经检查过，数据格式正常：

- 训练集数量：`50000`
- 测试集数量：`10000`
- 训练集每个类别各 `5000` 张
- 测试集每个类别各 `1000` 张
- 图像尺寸为 `32 x 32`
- 图像是 RGB 三通道彩色图
- 标签范围是 `0 ~ 9`

CIFAR-10 类别编号对应关系：

| 类别编号 | 类别名称 |
| --- | --- |
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

---

## 3. 训练

### 3.1 使用默认参数训练

在当前目录执行：

```bash
python train_cifar_10.py
```

默认参数为：

```text
epochs: 100
batch_size: 128
lr: 1e-3
data_root: /home/user/ubuntu/datasets/cifar-10-python
checkpoint: model_cifar_10.pt
num_samples: 10
sampling_steps: 250
sample_output: sample_cifar_10.png
```

训练过程中每个 epoch 结束后会保存一次模型到：

```text
model_cifar_10.pt
```

训练结束后会自动生成 10 张彩色图，保存到：

```text
sample_cifar_10.png
```

---

### 3.2 自定义训练参数

可以按需修改训练参数：

```bash
python train_cifar_10.py \
  --epochs 100 \
  --batch-size 128 \
  --lr 1e-3 \
  --data-root /home/user/ubuntu/datasets/cifar-10-python \
  --checkpoint model_cifar_10.pt \
  --num-samples 10 \
  --sampling-steps 250 \
  --sample-output sample_cifar_10.png
```

常用参数说明：

| 参数 | 说明 |
| --- | --- |
| `--epochs` | 训练轮数 |
| `--batch-size` | 每个 batch 的图像数量 |
| `--lr` | Adam 优化器学习率 |
| `--data-root` | CIFAR-10 数据集根目录 |
| `--checkpoint` | 模型保存路径 |
| `--num-samples` | 训练结束后生成图像的数量 |
| `--sampling-steps` | 采样时的 Euler 步数 |
| `--sample-output` | 生成图像保存路径 |

例如减少训练轮数，方便快速测试：

```bash
python train_cifar_10.py --epochs 1 --batch-size 128
```

如果想使用较大 batch size：

```bash
python train_cifar_10.py --batch-size 256
```

---

## 4. 推理 / 生成图像

推理脚本为：

```text
infer_cifar_10.py
```

它会加载训练得到的模型文件：

```text
model_cifar_10.pt
```

### 4.1 生成全部 10 个类别

直接运行：

```bash
python infer_cifar_10.py
```

默认行为：

- 生成 `10` 张图
- 每个类别生成一张
- 输出保存为 `sample_cifar_10.png`
- 采样步数为 `250`

---

### 4.2 生成指定类别

例如生成猫和狗，对应类别编号分别是 `3` 和 `7`：

```bash
python infer_cifar_10.py --labels 3 7
```

类别编号对应：

```text
0 airplane
1 automobile
2 bird
3 cat
4 deer
5 dog
6 frog
7 horse
8 ship
9 truck
```

例如只生成猫：

```bash
python infer_cifar_10.py --labels 3
```

生成飞机和轮船：

```bash
python infer_cifar_10.py --labels 0 8
```

---

### 4.3 修改输出文件名

```bash
python infer_cifar_10.py --labels 3 --output cat.png
```

---

### 4.4 修改采样步数

```bash
python infer_cifar_10.py --labels 3 --steps 500 --output cat.png
```

采样步数越大，生成过程通常越平滑，但速度也会更慢。

---

### 4.5 固定随机种子

如果希望多次生成结果一致，可以指定随机种子：

```bash
python infer_cifar_10.py --labels 3 --seed 42 --output cat_seed_42.png
```

---

### 4.6 指定模型路径

如果模型文件不使用默认名称，可以手动指定：

```bash
python infer_cifar_10.py \
  --checkpoint model_cifar_10.pt \
  --labels 3 \
  --output cat.png
```

---

## 5. 推理参数说明

| 参数 | 说明 |
| --- | --- |
| `--checkpoint` | 模型 checkpoint 路径 |
| `--labels` | 要生成的类别编号，可以传多个 |
| `--steps` | Euler 采样步数 |
| `--output` | 输出图片路径 |
| `--seed` | 随机种子 |

---

## 6. 可视化训练数据集

项目里提供了 `visualize_dataset.py`，用于查看 CIFAR-10 训练集，并从每个类别中取一张图片拼接成一张图。它的拼接方式与 `infer_cifar_10.py` 保持一致。

### 6.1 使用默认参数可视化

直接运行：

```bash
python visualize_dataset.py
```

默认行为：

- 使用训练集数据
- 从 `airplane` 到 `truck` 每个类别取一张图片
- 使用和推理结果相同的 `32 x 32` 图像尺寸
- 使用相同的 `padding=2`
- 按 `nrow=10` 拼接
- 输出保存为 `cifar10_train_examples.png`

输出示例路径：

```text
cifar10_train_examples.png
```

### 6.2 自定义可视化参数

可以修改输出文件路径、数据集路径：

```bash
python visualize_dataset.py \
  --data-root /home/user/ubuntu/datasets/cifar-10-python \
  --output dataset_samples.png
```

### 6.3 可视化参数说明

| 参数 | 说明 |
| --- | --- |
| `--data-root` | CIFAR-10 数据集根目录 |
| `--output` | 可视化结果保存路径 |

### 6.4 示例命令

使用默认路径并输出到指定文件：

```bash
python visualize_dataset.py --output cifar10_train_examples.png
```

---

## 7. 模型训练原理简述

Flow Matching 的目标是训练一个速度场：

```text
v(x_t, t, label)
```

其中：

- `x_0` 是随机噪声
- `x_1` 是真实图像
- `x_t` 是噪声和真实图像之间的插值结果
- `t` 表示时间，范围是 `[0, 1]`

训练时构造：

```text
x_t = (1 - t) * noise + t * image
```

模型需要预测：

```text
v = image - noise
```

损失函数为 MSE：

```text
loss = MSE(pred_v, image - noise)
```

推理时从噪声开始：

```text
x_0 = randn
```

然后用模型预测速度并逐步更新：

```text
x = x + v * dt
```

最终得到生成图像。

---

## 8. 生成结果说明

生成的图像是 RGB 彩色图，像素值会被映射回 `[0, 1]` 区间后保存为 PNG。

默认输出图像中，最多每行显示 10 张图。

例如使用：

```bash
python infer_cifar_10.py
```

会生成 10 张图，分别对应 CIFAR-10 的 10 个类别。

---

## 9. 常见问题

### 如果训练时报找不到数据集

请确认数据集路径是否正确：

```text
/home/user/ubuntu/datasets/cifar-10-python/cifar-10-batches-py
```

或者手动指定数据集路径：

```bash
python train_cifar_10.py --data-root /home/user/ubuntu/datasets/cifar-10-python
```

### 如果生成图像质量不好

可以尝试：

- 增加 `--epochs`
- 增大 `batch size`
- 降低学习率
- 增加推理时的 `--steps`
- 使用固定的 `--seed` 观察不同训练阶段的生成差异

### 如果显存不足

可以降低 batch size：

```bash
python train_cifar_10.py --batch-size 64
```

或者更低：

```bash
python train_cifar_10.py --batch-size 32
```
