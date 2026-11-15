"""Visualize one CIFAR-10 training image from each class.

The output grid intentionally matches the inference grid layout used by
``infer_cifar_10.py``: raw 32x32 RGB images, 10 images per row, and the same
default image padding.
"""

import argparse

import torch
import torchvision
import torchvision.transforms.functional as TF
import torchvision.utils


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
    parser = argparse.ArgumentParser(description="Visualize one CIFAR-10 training image per class.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/user/ubuntu/datasets/cifar-10-python",
        help="CIFAR-10 data root directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cifar10_train_examples.png",
        help="Output visualization file path.",
    )
    return parser.parse_args()


def find_one_image_per_class(dataset: torchvision.datasets.CIFAR10) -> torch.Tensor:
    """Find the first occurrence of each class and return a stacked tensor."""
    images = [None] * len(CLASS_NAMES)
    found = set()

    for image, label in dataset:
        if label not in found:
            images[label] = TF.pil_to_tensor(image).float() / 255.0
            found.add(label)
        if len(found) == len(CLASS_NAMES):
            break

    missing = [label for label, image in enumerate(images) if image is None]
    if missing:
        raise RuntimeError(f"Could not find images for labels: {missing}")

    return torch.stack(images)


def visualize(images: torch.Tensor, output_path: str) -> None:
    """Save a grid matching ``infer_cifar_10.py`` output layout."""
    torchvision.utils.save_image(
        images,
        output_path,
        nrow=min(10, images.size(0)),
        padding=2,
        pad_value=0.0,
    )


def main() -> None:
    args = parse_args()
    print(f"Loading CIFAR-10 training set from {args.data_root}...")

    dataset = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=False,
    )

    images = find_one_image_per_class(dataset)
    visualize(images, args.output)
    print(f"Saved visualization to {args.output}")
    print("Layout: 10 images per row, 32x32 images, padding=2 (matches inference).")


if __name__ == "__main__":
    main()
