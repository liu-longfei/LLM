# 推理：
## Unconditional Model
```
    device = "cuda"
    model = UNet().to(device)
    ckpt = torch.load("unconditional_ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=64, device=device)
    x = diffusion.sample(model, n=16)
    plot_images(x)
```
## Conditional Model
This model was trained on CIFAR-10 64x64 with 10 classes airplane:0, auto:1, bird:2, cat:3, deer:4, dog:5, frog:6, horse:7, ship:8, truck:9
```
    n = 10
    device = "cuda"
    model = UNet_conditional(num_classes=10).to(device)
    ckpt = torch.load("conditional_ema_ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=64, device=device)
    y = torch.Tensor([6] * n).long().to(device)
    x = diffusion.sample(model, n, y, cfg_scale=3)
    plot_images(x)
```

## 参考链接
### bilibili
https://www.bilibili.com/video/BV1we4y1H7gG/?spm_id_from=333.999.0.0&vd_source=fffd0a55d832fc312ff82010c51836f8

### youtube
https://www.youtube.com/watch?v=TBCRlnwJtZU

### code: 
https://github.com/dome272/Diffusion-Models-pytorch/tree/main

Sampling
The following examples show how to sample images using the models trained in the video on the Landscape Dataset. You can download the checkpoints for the models here.
