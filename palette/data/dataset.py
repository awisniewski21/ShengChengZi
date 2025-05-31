import glob
import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import IMG_EXTENSIONS


class Char2CharDataset(Dataset):
    def __init__(
        self,
        root_image_dir: str,
        image_size: int = 32,
    ):
        self.root_image_dir = Path(root_image_dir)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.samples = []
        for ext in IMG_EXTENSIONS:
            files_s = sorted(glob.glob(str(self.root_image_dir / f"**/Simplified/*{ext}"), recursive=True))
            files_t = sorted(glob.glob(str(self.root_image_dir / f"**/Traditional/*{ext}"), recursive=True))

            assert len(files_s) == len(files_t), "Number of simplified and traditional images must match"

            for f_s, f_t in zip(files_s, files_t):
                assert Path(f_s).stem == Path(f_t).stem, f"Image names must match: {f_s} vs {f_t}"
                self.samples.extend([{"src_file": f_t, "trg_file": f_s, "label": 0}]) # Traditional to Simplified (0)
                # self.samples.extend([{"src_file": f_s, "trg_file": f_t, "label": 1}]) # Simplified to Traditional (1)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_src = Image.open(self.root_image_dir / sample["src_file"])
        img_trg = Image.open(self.root_image_dir / sample["trg_file"])
        if self.transform:
            img_src = self.transform(img_src)
            img_trg = self.transform(img_trg)

        ret = {}
        ret['gt_image'] = img_trg
        ret['cond_image'] = img_src
        ret['path'] = os.path.basename(sample["trg_file"])
        return ret

    def __len__(self):
        return len(self.samples)