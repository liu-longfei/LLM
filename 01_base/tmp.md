```python
import torch
import torch.nn as nn


# 定义一个简单的全连接神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)  # 200 + 20 = 220
        self.fc2 = nn.Linear(20, 1)  # 20 + 1 + 21

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


# 创建模型实例
model = SimpleNet()

# 计算模型的参数量
total_params = sum(param.numel() for param in model.parameters())
print(model)
for name, param in model.named_parameters():
    print(f"参数名称: {name}, 参数类型: {param.dtype}, 参数形状: {param.shape}")
print(f"模型的参数量: {total_params}")

# 带有dropout和attention_mask的多头注意力写法
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_prob=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 定义线性层用于生成 Q, K, V
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, attention_mask=None):
        batch_size, seq_length, _ = x.size()

        # 计算 Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # 分割多头
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            # 扩展注意力掩码以匹配多头的维度
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 将掩码为 0 的位置设为负无穷，这样在 softmax 后对应的注意力权重就会趋近于 0
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))

        # 计算注意力概率
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 计算注意力输出
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)

        # 通过输出线性层
        output = self.out_proj(attn_output)
        return output


# 测试代码

# 定义模型参数
embed_dim = 512
num_heads = 8
batch_size = 16
seq_length = 32
dropout_prob = 0.1

# 创建模型实例
multihead_attn = MultiHeadAttention(embed_dim, num_heads, dropout_prob)

# 生成随机输入
x = torch.randn(batch_size, seq_length, embed_dim)

# 生成随机的注意力掩码
attention_mask = torch.randint(0, 2, (batch_size, seq_length))

# 前向传播
output = multihead_attn(x, attention_mask)

# 打印输出形状
print(f"输入形状: {x.shape}")
print(f"注意力掩码形状: {attention_mask.shape}")
print(f"输出形状: {output.shape}")

```
