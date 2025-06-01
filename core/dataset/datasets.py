import glob
import json
import random
from pathlib import Path
from typing import Callable, List, Tuple

from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision.datasets.folder import IMG_EXTENSIONS

from configs.base_config import TrainingConfigBase
from glyffuser.models import t5


class UnpairedImageDataset(Dataset):
    def __init__(
        self,
        root_image_dir: str | Path,
        metadata_path: str | Path | None = None,
        filename_label: str = "Filename",
        transform: Callable | None = None,
    ):
        self.root_image_dir = Path(root_image_dir)
        self.metadata_path = metadata_path
        self.filename_label = filename_label
        self.transform = transform

        self.samples = []
        if self.metadata_path is not None:
            with open(self.metadata_path, "r") as f:
                for line in f:
                    self.samples.append(json.loads(line))
        else:
            print(f"Metadata file not specified, so loading all images from '{self.root_image_dir}'...")
            for ext in IMG_EXTENSIONS:
                files = sorted(glob.glob(str(self.root_image_dir / f"**/*{ext}"), recursive=True))
                self.samples.extend([{self.filename_label: f} for f in files])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(self.root_image_dir / sample[self.filename_label])
        if self.transform:
            image = self.transform(image)
        return image


class UnpairedCaptionedImageDataset(UnpairedImageDataset):
    def __init__(
        self,
        *args,
        caption_label: str = "Chinese Definition",
        caption_encoder: str = "google-t5/t5-small",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.caption_label = caption_label
        self.caption_encoder = caption_encoder

    def __getitem__(self, idx: int):
        image = super().__getitem__(idx)

        caption = self.samples[idx][self.caption_label]
        caption_embed, caption_attn_mask = t5.t5_encode_text([caption], name=self.caption_encoder, return_attn_mask=True)
        return image, caption_embed.squeeze(), caption_attn_mask.squeeze()


class PairedImageDataset(Dataset):
    def __init__(
        self,
        root_image_dir: str | Path,
        metadata_path: str | Path | None = None,
        filename_label_s: str = "Filename (S)",
        filename_label_t: str = "Filename (T)",
        **kwargs,
    ):
        self.root_image_dir = Path(root_image_dir)
        self.metadata_path = metadata_path
        self.f_label_s = filename_label_s
        self.f_label_t = filename_label_t

        self.dataset_s = UnpairedImageDataset(root_image_dir, metadata_path=metadata_path, filename_label=self.f_label_s, **kwargs)
        self.dataset_t = UnpairedImageDataset(root_image_dir, metadata_path=metadata_path, filename_label=self.f_label_t, **kwargs)

        if metadata_path is None:
            self.dataset_s.samples = [s for s in self.dataset_s.samples if "Simplified" in s[self.f_label_s]]
            self.dataset_t.samples = [t for t in self.dataset_t.samples if "Traditional" in t[self.f_label_t]]

        for s, t in zip(self.dataset_s.samples, self.dataset_t.samples):
            f_s = s[self.f_label_s]
            f_t = t[self.f_label_t]
            assert Path(f_s).stem == Path(f_t).stem, f"Image names must match: '{f_s}' vs '{f_t}'"

    def __len__(self):
        return len(self.dataset_s)

    def __getitem__(self, idx: int):
        img_s = self.dataset_s[idx]
        img_t = self.dataset_t[idx]
        return img_t, img_s # Traditional to Simplified


class PairedBidirectionalImageDataset(PairedImageDataset):
    def __len__(self):
        return super().__len__() * 2 # Both src -> trg and trg -> src

    def __getitem__(self, idx: int):
        img_t, img_s = super().__getitem__(idx // 2)
        if idx % 2 == 0:
            return img_t, img_s, 0 # Traditional to Simplified (0)
        else:
            return img_s, img_t, 1 # Simplified to Traditional (1)


class ImageCollator:
    def __init__(self, task_name: str):
        self.task_name = task_name

    def __call__(self, batch_samples: List):
        if self.task_name != "text2char":
            return default_collate(batch_samples)

        images, texts, masks = zip(*batch_samples)
        texts = pad_sequence(texts, True)
        masks = pad_sequence(masks, True)
        batched_samples = list(zip(images, texts, masks))
        return default_collate(batched_samples)


def split_dataset(dataset: Dataset, validation_split: float | int, test_split: float | int = 0.0, 
                 shuffle: bool = True, seed: int = 0) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split a dataset into train, validation, and optionally test sets.
    
    Args:
        dataset: The dataset to split
        validation_split: Validation split (0.0-1.0 for percentage, or int for absolute count)
        test_split: Test split (0.0-1.0 for percentage, or int for absolute count)
        shuffle: Whether to shuffle the dataset before splitting
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset). test_dataset is None if test_split=0
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    if shuffle:
        random.seed(seed)
        random.shuffle(indices)
    
    # Calculate validation split size
    if isinstance(validation_split, float):
        if not 0.0 <= validation_split <= 1.0:
            raise ValueError("validation_split must be between 0.0 and 1.0 when using percentage")
        val_size = int(validation_split * dataset_size)
    else:
        val_size = validation_split
        if val_size < 0 or val_size >= dataset_size:
            raise ValueError(f"validation_split must be between 0 and {dataset_size-1} when using absolute count")
    
    # Calculate test split size
    if isinstance(test_split, float):
        if not 0.0 <= test_split <= 1.0:
            raise ValueError("test_split must be between 0.0 and 1.0 when using percentage")
        test_size = int(test_split * dataset_size)
    else:
        test_size = test_split
        if test_size < 0 or test_size >= dataset_size:
            raise ValueError(f"test_split must be between 0 and {dataset_size-1} when using absolute count")
    
    # Check that splits don't exceed dataset size
    if val_size + test_size >= dataset_size:
        raise ValueError(f"validation_split ({val_size}) + test_split ({test_size}) cannot exceed dataset size ({dataset_size})")
    
    # Create splits
    test_indices = indices[:test_size] if test_size > 0 else []
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices) if test_size > 0 else None
    
    return train_dataset, val_dataset, test_dataset


def get_dataloaders(cfg: TrainingConfigBase, *args, **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get train, validation, and test dataloaders with automatic splitting.
    
    Args:
        cfg: Training configuration containing split parameters
        *args, **kwargs: Arguments passed to the dataset constructor
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, test_dataloader). test_dataloader is None if test_split=0
    """
    cfg.task_name = cfg.task_name.lower()

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
    ])

    # Create full dataset
    if cfg.task_name == "rand2char":
        full_dataset = UnpairedImageDataset(*args, transform=transform, **kwargs)
    elif cfg.task_name == "text2char":
        full_dataset = UnpairedCaptionedImageDataset(*args, transform=transform, **kwargs)
    elif cfg.task_name == "char2char":
        full_dataset = PairedImageDataset(*args, transform=transform, **kwargs)
    elif cfg.task_name == "char2char_bi":
        full_dataset = PairedBidirectionalImageDataset(*args, transform=transform, **kwargs)
    else:
        raise ValueError(f"Unknown task name: '{cfg.task_name}'")

    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(
        full_dataset, 
        cfg.validation_split, 
        cfg.test_split, 
        cfg.shuffle_dataset, 
        cfg.seed
    )
    
    # Create dataloaders
    collate_fn = ImageCollator(cfg.task_name)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg.train_batch_size, 
        shuffle=cfg.shuffle_dataset, 
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=cfg.eval_batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    test_dataloader = None
    if test_dataset is not None:
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=cfg.eval_batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
    
    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset) if test_dataset else 0}")
    
    return train_dataloader, val_dataloader, test_dataloader


def get_dataloader(cfg: TrainingConfigBase, *args, **kwargs) -> DataLoader:
    """
    Backward-compatible function that returns only the training dataloader.
    For new code, use get_dataloaders() instead to get train/val/test splits.
    """
    train_dataloader, _, _ = get_dataloaders(cfg, *args, **kwargs)
    return train_dataloader
