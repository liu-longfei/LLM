"""Generate RGB CIFAR-10 images from a trained flow matching UNet."""

import argparse

import torch
import torchvision.utils

from train_cifar_10 import NUM_CLASSES, UNet


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CIFAR-10 images from a trained checkpoint.")
    parser.add_argument("--checkpoint", type=str, default="model_cifar_10.pt", help="Model checkpoint path.")
    parser.add_argument("--labels", type=int, nargs="+", default=None,
                        help="Class labels to generate. If omitted, generates all 10 classes.")
    parser.add_argument("--steps", type=int, default=250, help="Number of Euler sampling steps.")
    parser.add_argument("--output", type=str, default="sample_cifar_10.png", help="Output image path.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling.")
    return parser.parse_args()


def generate(
    model: UNet,
    labels: torch.Tensor,
    steps: int,
    output_path: str,
    device: torch.device,
) -> None:
    """Generate RGB samples with Euler integration from t=0 to t=1."""
    model.eval()
    num_images = labels.size(0)

    with torch.no_grad():
        x = torch.randn(num_images, 3, 32, 32, device=device)
        for step in range(steps):
            t = torch.full((num_images,), step / steps, device=device)
            velocity = model(x, t, labels)
            x = x + velocity / steps

    x = ((x.clamp(-1.0, 1.0) + 1.0) / 2.0).cpu()
    torchvision.utils.save_image(x, output_path, nrow=min(10, num_images))
    print(f"Saved {num_images} image(s) to {output_path}.")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.labels is None:
        labels = torch.arange(NUM_CLASSES, device=device)
    else:
        invalid_labels = [label for label in args.labels if label < 0 or label >= NUM_CLASSES]
        if invalid_labels:
            raise ValueError(f"Invalid labels: {invalid_labels}. Valid range is 0-{NUM_CLASSES - 1}.")
        labels = torch.tensor(args.labels, dtype=torch.long, device=device)

    model = UNet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint)

    print("Generating images for labels:")
    print(", ".join(f"{label.item()}={CLASS_NAMES[label.item()]}" for label in labels))
    generate(model, labels, args.steps, args.output, device)


if __name__ == "__main__":
    main()
