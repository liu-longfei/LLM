import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from model import UNet


# 1. 读取图片
image_path = "小火龙.jpeg"
image = Image.open(image_path)

# 2. 转换图片
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # 添加一个维度作为批处理维度

T = 1000

# 制定每一步的beta
betas = torch.linspace(1e-4, 0.02, T)

# 计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量
alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, 0)
sqrt_alphas_bar = torch.sqrt(alphas_bar)
sqrt_one_minus_alphas_bar = torch.sqrt(1. - alphas_bar)

assert alphas.shape == betas.shape == alphas_bar.shape == sqrt_alphas_bar.shape == sqrt_one_minus_alphas_bar.shape
print("all the same shape:", betas.shape)


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


t = torch.randint(T, size=(input_batch.shape[0], ), device=input_batch.device)
noise = torch.randn_like(input_batch)

x_t = (extract(sqrt_alphas_bar, t, input_batch.shape) * input_batch + extract(sqrt_one_minus_alphas_bar, t, input_batch.shape) * noise)

net_model = UNet(T=1000, ch=128, ch_mult=[1, 2, 3, 4], attn=[2], num_res_blocks=2, dropout=0.15)

epoch_num = 10000
optimizer = torch.optim.AdamW(net_model.parameters(), lr=1e-4, weight_decay=1e-4)
loss_init = float('inf')
for i in tqdm(range(epoch_num)):
    optimizer.zero_grad()
    loss = F.mse_loss(net_model(input_batch, t), noise)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net_model.parameters(), 1.)
    optimizer.step()
    print(loss)
    if loss < loss_init:
        loss_init = loss
        torch.save(net_model.state_dict(), 'ddpm.pt')
