import glob
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Tuple

from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision.datasets.folder import IMG_EXTENSIONS

from core.configs import TrainConfigBase
from glyffuser import t5


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

        raw_caption = self.samples[idx][self.caption_label]
        caption_embed, caption_attn_mask = t5.t5_encode_text([raw_caption], name=self.caption_encoder, return_attn_mask=True)
        return image, caption_embed.squeeze(), caption_attn_mask.squeeze(), raw_caption


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
    def __init__(self, dataset_task: str):
        self.dataset_task = dataset_task

    def __call__(self, batch_samples: List):
        if self.dataset_task == "text2char":
            images, texts_embed, texts_mask, raw_texts = zip(*batch_samples)
            texts_embed = pad_sequence(texts_embed, True)
            texts_mask = pad_sequence(texts_mask, True)
            batched_samples = list(zip(images, texts_embed, texts_mask, raw_texts))
            return default_collate(batched_samples)
        elif self.dataset_task == "char2char":
            return default_collate(batch_samples)
        else:
            return default_collate(batch_samples)


def get_dataset(cfg: TrainConfigBase, *args, **kwargs) -> Dataset:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
    ])

    cfg.dataset_task = cfg.dataset_task.lower()
    if cfg.dataset_task == "rand2char":
        full_dataset = UnpairedImageDataset(*args, transform=transform, **kwargs)
    elif cfg.dataset_task == "text2char":
        full_dataset = UnpairedCaptionedImageDataset(*args, transform=transform, **kwargs)
    elif cfg.dataset_task == "char2char":
        full_dataset = PairedImageDataset(*args, transform=transform, **kwargs)
    elif cfg.dataset_task == "char2char_bi":
        full_dataset = PairedBidirectionalImageDataset(*args, transform=transform, **kwargs)
    else:
        raise ValueError(f"Unknown dataset task name: '{cfg.dataset_task}'")

    return full_dataset


def get_dataloaders(cfg: TrainConfigBase, *args, full_dataset: Dataset | None = None, verbose: bool = True, **kwargs) -> Tuple[DataLoader, DataLoader | None, DataLoader | None]:
    """
    Get train, validation, and test dataloaders for the given task's dataset
    """
    if full_dataset is None:
        full_dataset = get_dataset(cfg, *args, **kwargs)

    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, cfg)

    collate_fn = ImageCollator(cfg.dataset_task)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collate_fn) if val_dataset is not None else None
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collate_fn) if test_dataset is not None else None

    if verbose:
        print("Dataset split:")
        print(f"    Train: {len(train_dataset)} images ({len(train_dataloader)} batches)")
        if val_dataset is not None:
            print(f"    Val: {len(val_dataset)} images ({len(val_dataloader)} batches)")
        if test_dataset is not None:
            print(f"    Test: {len(test_dataset)} images ({len(test_dataloader)} batches)")

    return train_dataloader, val_dataloader, test_dataloader


###
### Helper Functions
###

def split_dataset(dataset: Dataset, cfg: TrainConfigBase) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split a dataset into train, validation, and test sets
    Ensures that all samples for the same character are in the same split
    """
    dataset_size = len(dataset)

    def get_split_size(split_val):
        if isinstance(split_val, float):
            assert 0.0 <= split_val <= 1.0, "Data splits must be [0.0, 1.0] or [0, dataset_size]"
            return int(split_val * dataset_size)
        else:
            assert 0 <= split_val < dataset_size, "Data splits must be [0.0, 1.0] or [0, dataset_size]"
            return split_val if split_val > 0 else 0

    val_size = get_split_size(cfg.validation_split)
    test_size = get_split_size(cfg.test_split)
    assert val_size + test_size < dataset_size, "Validation and test splits must be less than dataset size"

    if not isinstance(dataset, PairedImageDataset):
        # Unpaired datasets - Each index is its own group
        all_ixs_grouped = [[ix] for ix in range(dataset_size)]
    else:
        # Paired datasets - Group indices so that all samples for the same character are in the same split
        # Assumes image file names are formatted as "<unicode>_<sequence>.png"
        all_ixs_grouped = defaultdict(list)
        for ix, sample in enumerate(dataset.dataset_s.samples):
            char_unicode_num = Path(sample[dataset.f_label_s]).stem.split("_")[0]
            all_ixs_grouped[char_unicode_num].extend([2*ix, 2*ix+1] if isinstance(dataset, PairedBidirectionalImageDataset) else [ix])
        all_ixs_grouped = list(all_ixs_grouped.values())

    random.seed(cfg.seed)
    random.shuffle(all_ixs_grouped)

    test_indices, val_indices, train_indices = [], [], []
    for ixs in all_ixs_grouped:
        if len(test_indices) < test_size:
            test_indices.extend(ixs)
        elif len(val_indices) < val_size:
            val_indices.extend(ixs)
        else:
            train_indices.extend(ixs)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_size > 0 else None
    test_dataset = Subset(dataset, test_indices) if test_size > 0 else None

    return train_dataset, val_dataset, test_dataset