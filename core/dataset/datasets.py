import glob
import json
from pathlib import Path
from typing import Callable, List

from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision.datasets.folder import IMG_EXTENSIONS

from core.config.base_config import TrainingConfigBase
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


def get_dataloader(cfg: TrainingConfigBase, *args, batch_size: int | None = None, shuffle: bool = True, **kwargs) -> DataLoader:
    cfg.task_name = cfg.task_name.lower()
    batch_size = batch_size if batch_size is not None else cfg.train_batch_size

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
    ])

    if cfg.task_name == "rand2char":
        dataset = UnpairedImageDataset(*args, transform=transform, **kwargs)
    elif cfg.task_name == "text2char":
        dataset = UnpairedCaptionedImageDataset(*args, transform=transform, **kwargs)
    elif cfg.task_name == "char2char":
        dataset = PairedImageDataset(*args, transform=transform, **kwargs)
    elif cfg.task_name == "char2char_bi":
        dataset = PairedBidirectionalImageDataset(*args, transform=transform, **kwargs)
    else:
        raise ValueError(f"Unknown task name: '{cfg.task_name}'")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=ImageCollator(cfg.task_name))
