import numpy as np
import torch
import tqdm


## The number of sampling steps.
num_steps = 100


def euler_sampler(score_model,
                  marginal_prob_std,
                  diffusion_coeff,
                  batch_size=64,
                  num_steps=num_steps,
                  device='cuda',
                  eps=1e-3):
    # Step 1 定义初始时间1和先验分布的随机样本
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]

    # Step 2 定义采样的逆时间网格以及每一步的时间步长
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]

    # Step 3 根据欧拉算法来求解逆时间SDE
    x = init_x
    with torch.no_grad():
        for time_step in tqdm.tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device)* time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)

    # Step 4 取最后一步的期望值作为生成的样本
    return mean_x


signal_to_noise_ratio = 0.16
num_steps = 500


def pc_sampler(score_model,
               marginal_prob_std,
               diffusion_coeff,
               batch_size = 64,
               num_steps=num_steps,
               snr=signal_to_noise_ratio, # 多出来的一项
               device='cuda',
               eps=1e-3):
    """ Generate samples from score-based models with Predictor-Corrector method.

    Args:
        score_model: A Pytorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that gives the standard deviation of the perturbation kernel.
        diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
        batch_size: The number of sampling steps.Equivalent to the number of discretied time steps.
        device: 'cuda' for running on GPUs,and 'cpu' for running on GPUs.
        eps: The smallest time step for numerical stability.

    Returns:
        Samples.
    """
    # Step1 定义初始时间1和先验分布的随机样本
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]

    # Step 2 定义采样的逆时间网格以及每一步的时间步长
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]

    # Step 3 重复交替的进行郎之万采样和逆时间SDE的欧拉数值求解
    x = init_x
    with torch.no_grad():
        for time_step in tqdm.tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step

            # Corrector step (Lang in MCMC)
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
            print(f"{langevin_step_size=}")

            for _ in range(10):
                x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x) # 郎之万采样迭代公式
                grad = score_model(x, batch_time_step)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = np.sqrt(np.prod(x.shape[1:]))
                langevin_step_size = 2 * (snr * noise_norm / grad_norm)**2
                print(f"{langevin_step_size=}")

            # Predictor step (Euler-Maruyama)
            g = diffusion_coeff(batch_time_step)
            x_mean = x + (g**2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = x_mean + torch.sqrt(g**2 * step_size)[:, None, None, None] * torch.randn_like(x)

        # Step 4 取最后一步的欧拉数值求解的期望值作为最终生成的样本
        return x_mean


from scipy import integrate


# The error tolerance for the black-box ODE solver
error_tolerance = 1e-5


def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                batch_size = 64,
                atol = error_tolerance,
                rtol = error_tolerance,
                device='cuda',
                z=None,
                eps=1e-3):

    # Step 1 定义初始时间1和初始值x
    t = torch.ones(batch_size, device=device)
    if z is None:
        init_x = torch.randn(batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
    else:
        init_x = z

    shape = init_x.shape

    # Step 2 定义分数预测函数和常微分函数
    def score_eval_wrapper(sample, time_steps):
        """ A wrapper to the score-base model for use by the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
        with torch.no_grad():
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshpae((-1,)).astype(np.float16)

    def ode_func(t, x):
        """ The ODE function for use by the ODE solver."""
        time_steps = np.ones((shape[0],)) * t
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return -0.5 * (g**2) * score_eval_wrapper(x, time_steps)

    # Step 3 调用常微分求解算子来解出t=eps时刻的值，即预测的样本
    res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
    print(f"Number of function evaluation: {res.nfev}")

    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

    return x


from torchvision.utils import make_grid
import time
from score_base_model import ScoreNet
from score_base_model import marginal_prob_std_fn
from score_base_model import diffusion_coeff_fn

# Load the pre_trained checkpoint from disk.
device = 'cpu'
ckpt = torch.load('model/ckpt.pth', map_location=device)
score_model = ScoreNet
score_model.load_state_dict(ckpt)

sample_batch_size = 64
sampler = pc_sampler  # 可以是其中之一['euler_sampler', 'pc_sampler', 'ode_sampler']
"""
pc_sampler:采样耗时12.482401132583618s
euler_sampler:采样耗时0.300966739654541s
ode_sampler:
"""
t1 = time.time()
# Generate samples using the specified sampler.
samples = sampler(score_model,
                  marginal_prob_std_fn,
                  diffusion_coeff_fn,
                  sample_batch_size,
                  device=device)
t2 = time.time()
print(f"{str(sampler)}采样耗时{t2-t1}s")

# Sample visualization
samples = samples.clamp(0.0, 1.0)


import matplotlib.pyplot as plt


sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

plt.figure(figsize=(6, 6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()
