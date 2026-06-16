"""Image classification datasets for HW3.5 ablation."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


class CropClassificationDataset(Dataset):
    """Load real crops from ImageFolder-style layout."""

    def __init__(
        self,
        root: Path,
        class_to_idx: dict[str, int],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.root = Path(root)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for class_name, class_idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class MixedClassificationDataset(Dataset):
    """Real train crops + synthetic images for selected classes."""

    def __init__(
        self,
        real_root: Path,
        synthetic_root: Path,
        synthetic_classes: tuple[str, ...],
        class_to_idx: dict[str, int],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.real = CropClassificationDataset(real_root, class_to_idx, transform=None)
        self.synthetic = CropClassificationDataset(
            synthetic_root,
            {name: class_to_idx[name] for name in synthetic_classes if name in class_to_idx},
            transform=None,
        )
        self.transform = transform
        self.samples = list(self.real.samples) + list(self.synthetic.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label
