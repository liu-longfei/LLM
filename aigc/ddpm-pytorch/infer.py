import os
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from modules import UNet
# 直接复制你代码里的 Diffusion 类放这里，或者import
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        print(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


if __name__ == "__main__":
    # ===================== 配置区，自己改 =====================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 64        # 和训练保持一致！你训练用的是64
    ckpt_path = "./models/DDPM_Uncondtional/ckpt.pt"  # 你的权重路径
    save_path = "./single_sample.png"
    num_sample = 1       # 1就是单张图，改成8就是8张
    # =========================================================

    # 初始化模型 + 扩散
    model = UNet().to(device)
    diffusion = Diffusion(img_size=img_size, device=device)

    # 加载权重
    print(f"loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)

    # 采样
    sampled_imgs = diffusion.sample(model, n=num_sample)  # [1,3,64,64]

    # 保存单张图片
    img_tensor = sampled_imgs[0]  # 取第一张
    # 转成plt格式 (H,W,3)
    img_np = img_tensor.cpu().permute(1,2,0).numpy()

    plt.imsave(save_path, img_np)
    print(f"图片已保存至: {save_path}")

    # 可选：弹出图片查看
    plt.figure()
    plt.imshow(img_np)
    plt.axis("off")
    plt.show()
