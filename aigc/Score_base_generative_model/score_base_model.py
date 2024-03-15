import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from copy import deepcopy

class TimeEncoding(nn.Module):
    """用于对时间进行特定傅里叶编码"""
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    """基于U-net的时间依赖的分数估计模型"""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
            marginal_prob_std:A function that takes time t and gives the standard
            deviation of the preturbation kernel p_{0t}(x(t) | x(0)).
            channels: The number of channels for feature maps of each resolution.
            embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(TimeEncoding(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))  # 时间编码层
        # U-net 的编码器部分，空间不断减小，通道不断增大
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)  # to skip connection
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)  # to skip connection
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)  # to skip connection
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)  # to skip connection
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # U-net的解码器部分，空间不断增大，通道不断减小，并且有来自编码器部分的skip connection
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        # 对时间t进行编码
        embed = self.act(self.embed(t))

        # 编码器部分前向计算
        h1 = self.conv1(x)
        h1 += self.dense1(embed)  # 注入时间t
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)  # 注入时间t
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)  # 注入时间t
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)  # 注入时间t
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # 解码部分前向计算
        h = self.tconv4(h4)
        h += self.dense5(embed)  # 注入时间t
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))  # skip connection
        h += self.dense6(embed)  # 注入时间t
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))  # skip connection
        h += self.dense7(embed)  # 注入时间t
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))  # skip connection

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]  # 目的是希望预测的分数的二阶范数逼近与真实分数的二阶范数

        return h


device = 'cpu'


def marginal_prob_std(t, sigma):
    """计算任意t时刻的扰动后条件高斯分布的标准差"""
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma):
    """ 计算任意t时刻的扩散系数，本例定义的SDE没有漂移系数"""
    return torch.tensor(sigma**t, device=device)


sigma = 25.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)  # 构建无参函数
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)  # 构建无参函数


def loss_fn(score_model, x, marginal_prob_std, eps=1e-5):
    """ The loss function for training score-based generative models.

    Args:
        model: A Pytorch model instance that represents a
        time-dependent score-based model.
        x: A mini-batch of training data.
        marginal_prob_std: A function that gives the standard deviation of the perturbation kernel.
        eps: A tolerance value for numerical stability.
    """
    # Step 1 从[0.00001, 0.9999]中随机生成batchsize个浮点型t
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps

    # Step 2基于重参数技巧采样出分布p_t(x)的一个随机样本preturbed_x
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]

    # Step 3 将当前的加噪样本和时间输入到Score Netword中预测出分数Score
    score = score_model(perturbed_x, random_t)

    # Step 4 计算score matching loss
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1, 2, 3)))
    return loss


class EMA(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.model = deepcopy(model)
        self.model.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.model.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import tqdm


score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

n_epochs = 50
batch_size = 32
lr = 1e-4

dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

optimizer = Adam(score_model.parameters(), lr=lr)
tqdm_epoch = tqdm.tqdm(range(n_epochs))

ema = EMA(score_model)
for epoch in tqdm_epoch:
    avg_loss = 0
    num_items = 0
    for x, y in data_loader:
        x = x.to(device)
        loss = loss_fn(score_model, x, marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(score_model)
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]

    print('Average ScoreMatching Loss: {:5f}'.format(avg_loss / num_items))
    torch.save(score_model.state_dict(), 'model/ckpt.pth')
