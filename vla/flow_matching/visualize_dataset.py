"""Visualize one CIFAR-10 training image from each class in a single grid."""

import argparse

import matplotlib.pyplot as plt
import torchvision


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
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output image DPI.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot after saving it.",
    )
    return parser.parse_args()


def find_one_image_per_class(dataset: torchvision.datasets.CIFAR10):
    """Find the first occurrence of each class and return (images, labels)."""
    images = [None] * len(CLASS_NAMES)
    found = set()

    for image, label in dataset:
        if label not in found:
            images[label] = image
            found.add(label)
        if len(found) == len(CLASS_NAMES):
            break

    missing = [label for label, image in enumerate(images) if image is None]
    if missing:
        raise RuntimeError(f"Could not find images for labels: {missing}")

    return images


def visualize(images, output_path: str, dpi: int) -> None:
    """Save a 2x5 visualization containing one image from every class."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.5))

    for label, image in enumerate(images):
        ax = axes[label // 5][label % 5]
        ax.imshow(image)
        ax.set_title(f"{label}: {CLASS_NAMES[label]}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    print(f"Loading CIFAR-10 training set from {args.data_root}...")

    dataset = torchvision.datasets.CIFAR10(
        root=args.data_root,
        train=True,
        download=False,
    )

    images = find_one_image_per_class(dataset)
    visualize(images, args.output, args.dpi)
    print(f"Saved visualization to {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
