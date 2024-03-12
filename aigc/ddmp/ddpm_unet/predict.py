import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from model import UNet


T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1. - betas
alphas_bar = torch.cumprod(alphas, dim=0)
alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
coeff1 = torch.sqrt(1. / alphas)
coeff2 = coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar)
posterior_var = betas * (1. - alphas_bar_prev) / (1. - alphas_bar)

net_model = UNet(T=1000, ch=128, ch_mult=[1, 2, 3, 4], attn=[2], num_res_blocks=2, dropout=0.15)
ckpt = torch.load('ddpm.pt')
net_model.load_state_dict(ckpt)
print("model load weight done.")
net_model.eval()

noisyImage = torch.randn(size=[1, 3, 32, 32])
saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
save_image(saveNoisy, 'noise.png', nrow=8)


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def predict_xt_prev_mean_from_eps(x_t, t, eps):
    assert x_t.shape == eps.shape
    return (extract(coeff1, t, x_t.shape) * x_t - extract(coeff2, t, x_t.shape) * eps)


def p_mean_variance(x_t, t):
    # below: only log_variance is used in the KL computations
    var = torch.cat([posterior_var[1:2], betas[1:]])
    var = extract(var, t, x_t.shape)

    eps = net_model(x_t, t)
    xt_prev_mean = predict_xt_prev_mean_from_eps(x_t, t, eps=eps)

    return xt_prev_mean, var


x_t = x_T = noisyImage
with torch.no_grad():
    for time_step in reversed(range(T)):
        print(time_step)
        t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
        mean, var = p_mean_variance(x_t=x_t, t=t)
        # no noise when t == 0
        if time_step > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = 0
        x_t = mean + torch.sqrt(var) * noise
        assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
x_0 = x_t
x_0 = torch.clip(x_0, -1, 1)
sampledImgs = x_0 * 0.5 + 0.5  # [0 ~ 1]

save_image(sampledImgs, 'sample.png', nrow=8)
