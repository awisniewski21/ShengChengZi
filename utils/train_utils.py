import glob
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS

from config.base_config import TrainingConfigBase
from models import t5
from utils.utils import get_repo_dir


class ImageDataset(Dataset):
    def __init__(
        self,
        root_image_dir: str | Path,
        filename_label: str = "Filename",
        transform: Callable | None = None,
        caption_jsonl_path: str | Path | None = None,
        caption_label: str | None = None,
        caption_encoder: str | None = None,
    ):
        self.root_image_dir = Path(root_image_dir)
        self.filename_label = filename_label
        self.transform = transform
        self.caption_jsonl_path = caption_jsonl_path
        self.caption_label = caption_label
        self.caption_encoder = caption_encoder

        self.samples = []
        if self.caption_jsonl_path is not None:
            with open(self.caption_jsonl_path, "r") as f:
                for line in f:
                    self.samples.append(json.loads(line))
        else:
            for ext in IMG_EXTENSIONS:
                files = glob.glob(str(self.root_image_dir / f"**/*{ext}"), recursive=True)
                self.samples.extend([{self.filename_label: f} for f in files])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(self.root_image_dir / sample[self.filename_label])
        if self.transform:
            image = self.transform(image)

        if self.caption_jsonl_path is not None:
            caption = sample[self.caption_label]
            caption_embed, caption_attn_mask = t5.t5_encode_text([caption], name=self.caption_encoder, return_attn_mask=True)
            return image, caption_embed.squeeze(), caption_attn_mask.squeeze()
        else:
            return image


class PairedImageDataset(Dataset):
    def __init__(
        self,
        root_image_dir: str | Path,
        transform: Callable | None = None,
    ):
        self.root_image_dir = Path(root_image_dir)
        self.transform = transform

        self.samples = []
        for ext in IMG_EXTENSIONS:
            files_s = sorted(glob.glob(str(self.root_image_dir / f"**/Simplified/*{ext}"), recursive=True))
            files_t = sorted(glob.glob(str(self.root_image_dir / f"**/Traditional/*{ext}"), recursive=True))

            assert len(files_s) == len(files_t), "Number of simplified and traditional images must match"

            for f_s, f_t in zip(files_s, files_t):
                assert Path(f_s).stem == Path(f_t).stem, f"Image names must match: {f_s} vs {f_t}"
                self.samples.extend([{"src_file": f_t, "trg_file": f_s, "label": 0}]) # Traditional to Simplified (0)
                self.samples.extend([{"src_file": f_s, "trg_file": f_t, "label": 1}]) # Simplified to Traditional (1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_src = Image.open(self.root_image_dir / sample["src_file"])
        img_trg = Image.open(self.root_image_dir / sample["trg_file"])
        if self.transform:
            img_src = self.transform(img_src)
            img_trg = self.transform(img_trg)

        return img_src, img_trg, sample["label"]


class ImageCollator:
    def __init__(self, use_caption: bool = False):
        self.use_caption = use_caption

    def __call__(self, batch_samples: List):
        if not self.use_caption:
            return default_collate(batch_samples)

        images, texts, masks = zip(*batch_samples)
        texts = pad_sequence(texts, True)
        masks = pad_sequence(masks, True)
        batched_samples = list(zip(images, texts, masks))
        return default_collate(batched_samples)


def get_dataloader(
    cfg: TrainingConfigBase,
    root_image_dir: str | Path,
    filename_label: str = "Filename",
    caption_jsonl_path: str | Path | None = None,
    caption_label: str | None = None,
    caption_encoder: str | None = None,
) -> DataLoader:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(
        root_image_dir=root_image_dir,
        filename_label=filename_label,
        transform=transform,
        caption_jsonl_path=caption_jsonl_path,
        caption_label=caption_label,
        caption_encoder=caption_encoder,
    )

    return DataLoader(dataset, batch_size=cfg.train_batch_size, shuffle=True, collate_fn=ImageCollator(use_caption=caption_jsonl_path is not None))

def get_paired_dataloader(
    cfg: TrainingConfigBase,
    root_image_dir: str | Path,
) -> DataLoader:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
    ])

    dataset = PairedImageDataset(
        root_image_dir=root_image_dir,
        transform=transform,
    )

    return DataLoader(dataset, batch_size=cfg.train_batch_size, shuffle=True)