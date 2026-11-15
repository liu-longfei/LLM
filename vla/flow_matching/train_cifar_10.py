"""Flow matching conditional UNet training and sampling on CIFAR-10.

Compared with ``train.py``, this script uses the RGB CIFAR-10 dataset and
trains/generates 3-channel 32x32 color images.
"""

import argparse
import os

import torch
import torch.nn.functional as F
import torchvision
import torchvision.utils
from torch import Tensor, nn


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

NUM_CLASSES = 10
NUM_CHANNELS = 3
IMAGE_SIZE = 32
LABEL_EMB_DIM = 16
TIME_EMB_DIM = 16
TIME_HIDDEN_DIM = 32
CHANNELS = (64, 128, 256)
DATA_ROOT = "/home/user/ubuntu/datasets/cifar-10-python"
CHECKPOINT_PATH = "model_cifar_10.pt"
CHECKPOINT_TMP_PATH = ".model_cifar_10.pt"
SAMPLE_PATH = "sample_cifar_10.png"


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def add_condition(tensor: Tensor, cond_emb: Tensor) -> Tensor:
    """Broadcast a per-sample condition embedding onto a feature map."""
    cond_map = cond_emb.view(cond_emb.size(0), cond_emb.size(1), 1, 1)
    cond_map = cond_map.expand(-1, -1, tensor.size(2), tensor.size(3))
    return torch.cat([tensor, cond_map], dim=1)


def build_transform() -> torchvision.transforms.Compose:
    """Build the CIFAR-10 transform: ToTensor and scale to [-1, 1]."""
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: 2.0 * x - 1.0),
    ])


# --------------------------------------------------------------------------- #
# Model components
# --------------------------------------------------------------------------- #

class DownSample(nn.Module):
    """Downsampling block: conv layers followed by max pooling."""

    def __init__(self, in_channels: int, out_channels: int, condition_size: int):
        super().__init__()
        self.downconv = nn.Sequential(
            nn.Conv2d(in_channels + condition_size, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: Tensor, cond_emb: Tensor) -> tuple[Tensor, Tensor]:
        x = self.downconv(add_condition(x, cond_emb))
        return x, self.maxpool(x)


class UpSample(nn.Module):
    """Upsampling block: transposed conv followed by conv layers with skip."""

    def __init__(self, in_channels: int, out_channels: int, condition_size: int):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels + condition_size, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.upconv = nn.Sequential(
            nn.Conv2d(in_channels + condition_size, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: Tensor, residual_x: Tensor, cond_emb: Tensor) -> Tensor:
        x = self.deconv(add_condition(x, cond_emb))
        x = torch.cat([x, residual_x], dim=1)
        x = self.upconv(add_condition(x, cond_emb))
        return x


class UNet(nn.Module):
    """Conditional U-Net for flow matching velocity prediction on RGB images."""

    def __init__(self):
        super().__init__()
        self.condition_size = LABEL_EMB_DIM + TIME_EMB_DIM

        self.label_emb = nn.Embedding(num_embeddings=NUM_CLASSES, embedding_dim=LABEL_EMB_DIM)
        self.t_emb = nn.Sequential(
            nn.Linear(1, TIME_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(TIME_HIDDEN_DIM, TIME_EMB_DIM),
        )

        c1, c2, c3 = CHANNELS
        self.down0 = DownSample(in_channels=NUM_CHANNELS, out_channels=c1, condition_size=self.condition_size)
        self.down1 = DownSample(in_channels=c1, out_channels=c2, condition_size=self.condition_size)
        self.down2 = DownSample(in_channels=c2, out_channels=c3, condition_size=self.condition_size)

        self.up0 = UpSample(in_channels=c3, out_channels=c2, condition_size=self.condition_size)
        self.up1 = UpSample(in_channels=c2, out_channels=c1, condition_size=self.condition_size)

        self.output_conv = nn.Conv2d(c1 + self.condition_size, NUM_CHANNELS, kernel_size=3, padding=1)

    def forward(self, x: Tensor, t: Tensor, label: Tensor) -> Tensor:
        """Predict velocity v(x_t, t, label).

        Args:
            x: Noisy RGB images, shape (B, 3, 32, 32).
            t: Flow matching time steps in [0, 1], shape (B,).
            label: Class labels, shape (B,).

        Returns:
            Predicted RGB velocity, shape (B, 3, 32, 32).
        """
        cond_emb = torch.cat([self.label_emb(label), self.t_emb(t.unsqueeze(1))], dim=1)

        x0, x = self.down0(x, cond_emb)   # (B, c1, 32, 32) -> (B, c1, 16, 16)
        x1, x = self.down1(x, cond_emb)   # (B, c2, 16, 16) -> (B, c2, 8, 8)
        x, _ = self.down2(x, cond_emb)    # (B, c3, 8, 8)   -> (B, c3, 4, 4)
        x = self.up0(x, x1, cond_emb)     # (B, c2, 8, 8)
        x = self.up1(x, x0, cond_emb)     # (B, c1, 16, 16)
        return self.output_conv(add_condition(x, cond_emb))


# --------------------------------------------------------------------------- #
# Training and generation
# --------------------------------------------------------------------------- #

def train_one_epoch(
    model: UNet,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return the average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_img, batch_labels in dataloader:
        batch_img = batch_img.to(device)
        batch_labels = batch_labels.to(device)
        batch_size = batch_img.size(0)

        # Interpolate from noise (t=0) to data (t=1).
        batch_t = torch.rand(batch_size, device=device)
        batch_noise = torch.randn_like(batch_img)
        t_view = batch_t.view(-1, 1, 1, 1)
        batch_xt = (1.0 - t_view) * batch_noise + t_view * batch_img

        pred_vt = model(batch_xt, batch_t, batch_labels)
        loss = F.mse_loss(pred_vt, batch_img - batch_noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def save_checkpoint(model: UNet, path: str) -> None:
    """Atomically save the model state dict."""
    torch.save(model.state_dict(), CHECKPOINT_TMP_PATH)
    os.replace(CHECKPOINT_TMP_PATH, path)


def generate_color_images(
    model: UNet,
    device: torch.device,
    num_images: int,
    steps: int,
    output_path: str,
) -> None:
    """Generate and save a grid of conditional RGB CIFAR-10 samples.

    The first 10 images correspond to classes 0..9. If more images are
    requested, labels repeat cyclically.
    """
    model.eval()
    labels = torch.arange(num_images, device=device) % NUM_CLASSES

    with torch.no_grad():
        x = torch.randn(num_images, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)
        for step in range(steps):
            t = torch.full((num_images,), step / steps, device=device)
            velocity = model(x, t, labels)
            x = x + velocity / steps

    x = ((x.clamp(-1.0, 1.0) + 1.0) / 2.0).cpu()
    torchvision.utils.save_image(x, output_path, nrow=min(10, num_images))
    print(f"Saved {num_images} color samples to {output_path}.")


def load_model_state(model: UNet, checkpoint_path: str) -> None:
    """Load a plain state dict or a checkpoint containing model_state_dict."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train conditional UNet with flow matching on CIFAR-10.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT, help="CIFAR-10 data root directory containing cifar-10-batches-py.")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, help="Checkpoint save/load path.")
    parser.add_argument("--init_checkpoint", type=str, default='./model_cifa_10.pt', help="Initialize model weights from this checkpoint and continue training.")
    parser.add_argument("--generate", action="store_true", help="Skip training and generate from the checkpoint.")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of images to generate.")
    parser.add_argument("--sampling-steps", type=int, default=250, help="Number of Euler sampling steps.")
    parser.add_argument("--sample-output", type=str, default=SAMPLE_PATH, help="Generated sample image path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet().to(device)

    if args.generate:
        load_model_state(model, args.checkpoint)
        generate_color_images(model, device, args.num_samples, args.sampling_steps, args.sample_output)
        return

    if args.init_checkpoint:
        if not os.path.isfile(args.init_checkpoint):
            raise FileNotFoundError(f"Init checkpoint not found: {args.init_checkpoint}")
        load_model_state(model, args.init_checkpoint)
        print(f"Initialized model weights from {args.init_checkpoint}.")

    print(f"Loading CIFAR-10 RGB dataset from {args.data_root}...")
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_root, train=True, download=False, transform=build_transform(),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device)
        save_checkpoint(model, args.checkpoint)
        print(f"epoch={epoch:03d}  avg_loss={avg_loss:.6f}")

    print(f"Training complete. Checkpoint saved to {args.checkpoint}.")
    generate_color_images(model, device, args.num_samples, args.sampling_steps, args.sample_output)


if __name__ == "__main__":
    main()
