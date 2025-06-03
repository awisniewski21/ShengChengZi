import glob
import os
import random
import re
from pathlib import Path
from typing import Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm

from configs import TrainConfigBase


def load_unihan_data(unihan_dir: Path) -> pd.DataFrame:
    """
    Load and process Unihan data from text files
    """
    long_dfs_list = []
    for file_path in sorted(glob.glob(str(unihan_dir / "*.txt"))):
        this_df_long = pd.read_table(file_path, comment="#", names=["Unicode", "Key", "Value"])
        long_dfs_list.append(this_df_long)
    df_long = pd.concat(long_dfs_list, ignore_index=True)

    # Pivot to get one row per character
    df_full = df_long.pivot(index="Unicode", columns="Key", values="Value").reset_index()

    df_full["Unicode Int"] = df_full["Unicode"].map(lambda val: int(str(val).removeprefix("U+"), 16))
    df_full["Character"] = df_full["Unicode Int"].map(chr)
    df_full = df_full.sort_values("Unicode Int", ignore_index=True)
    df_full = df_full.set_index(sorted(c for c in df_full.columns if not c.startswith("k"))).reset_index()

    return df_full


def gen_dataset(
    df_full: pd.DataFrame,
    dataset_type: str,
    out_dir: Path,
    font_dir: Path,
    image_size: Tuple[int, int],
    font_size: int,
):
    """
    Generate dataset for all fonts
    """
    assert dataset_type in ["unpaired", "paired"], "dataset_type must be either 'unpaired' or 'paired'"
    is_unpaired = dataset_type == "unpaired"

    print(f"Generating {dataset_type} dataset...")
    data_df = get_unpaired_dataset(df_full) if is_unpaired else get_paired_dataset(df_full)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dfs_list = []
    for font_path in sorted(glob.glob(str(font_dir / "*.ttf"))):
        print(f"Generating images of characters for font '{Path(font_path).stem}'...")
        this_out_df = gen_char_images(data_df, dataset_type, out_dir, font_path, image_size, font_size)
        out_dfs_list.append(this_out_df)

    out_df = pd.concat(out_dfs_list, ignore_index=True)
    with open(out_dir / "metadata.jsonl", "w") as f:
        f.write(out_df.to_json(orient="records", lines=True))

    num_images = len(out_df) * (1 if is_unpaired else 2)
    print(f"Generated {dataset_type} dataset of {num_images:,} total images")


def gen_char_images(
    df: pd.DataFrame,
    dataset_type: str,
    out_dir: str | Path,
    font_path: str | Path,
    image_size: Tuple[int, int],
    font_size: int,
):
    """
    Generate and save images for each pair of characters in the DataFrame
    """
    is_unpaired = dataset_type == "unpaired"
    font = ImageFont.truetype(font_path, font_size)
    font_name = Path(font_path).stem

    rel_out_dirs = [Path(font_name)] if is_unpaired else [Path(font_name) / f for f in ["Simplified", "Traditional"]]
    out_dirs = [Path(out_dir) / rel_o for rel_o in rel_out_dirs]
    for o in out_dirs:
        os.makedirs(o, exist_ok=True)

    invalid_ixs = []
    for ix, row in tqdm(df.iterrows(), total=len(df)):
        filenames = [o / row["Filename"] for o in out_dirs]

        if any(os.path.exists(f) for f in filenames):
            assert all(os.path.exists(f) for f in filenames)
            continue

        chars = [row["Character"]] if is_unpaired else [row["Character (S)"], row["Character (T)"]]
        if any(not is_valid_char(c, font, image_size) for c in chars):
            invalid_ixs.append(ix)
            continue

        for c, f in zip(chars, filenames):
            img = create_image(c, font, image_size)
            img.save(f)

    out_df = df.drop(index=invalid_ixs)

    filename_cols = ["Filename"] if is_unpaired else ["Filename (S)", "Filename (T)"]
    for f_col, rel_o in zip(filename_cols, rel_out_dirs):
        out_df[f_col] = out_df["Filename"].map(lambda val: str(rel_o / val))

    out_df = out_df[sorted(c for c in out_df.columns if not c.startswith("k"))]

    return out_df


def split_dataset(dataset: Dataset, cfg: TrainConfigBase) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split a dataset into train, validation, and test sets
    """
    dataset_size = len(dataset)
    all_indices = list(range(dataset_size))

    random.seed(cfg.seed)
    random.shuffle(all_indices)

    if isinstance(cfg.validation_split, float):
        assert 0.0 <= cfg.validation_split <= 1.0, "validation_split must be [0.0, 1.0] or [0, dataset_size]"
        val_size = int(cfg.validation_split * dataset_size)
    else:
        assert 0 <= cfg.validation_split < dataset_size, "validation_split must be [0.0, 1.0] or [0, dataset_size]"
        val_size = cfg.validation_split

    if isinstance(cfg.test_split, float):
        assert 0.0 <= cfg.test_split <= 1.0, "test_split must be [0.0, 1.0] or [0, dataset_size]"
        test_size = int(cfg.test_split * dataset_size)
    else:
        assert 0 <= cfg.test_split < dataset_size, "test_split must be [0.0, 1.0] or [0, dataset_size]"
        test_size = cfg.test_split

    assert val_size + test_size < dataset_size, "Validation and test splits must be less than dataset size"

    test_indices = all_indices[:test_size] if test_size > 0 else []
    val_indices = all_indices[test_size:test_size + val_size] if val_size > 0 else []
    train_indices = all_indices[test_size + val_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_size > 0 else None
    test_dataset = Subset(dataset, test_indices) if test_size > 0 else None

    return train_dataset, val_dataset, test_dataset


###
### Helper Functions
###

def get_unpaired_dataset(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare unpaired Chinese character dataset with definitions
    """
    df = df_full.copy()

    # Get clean definitions
    df["Chinese Definition"] = df["kDefinition"].map(clean_definition, na_action="ignore")
    df = df.query("`Chinese Definition`.str.len() > 0").copy()

    df["Filename"] = df["Unicode Int"].map(lambda val: f"{val}.png")
    df = df.set_index(sorted(c for c in df.columns if not c.startswith("k"))).reset_index()

    return df


def get_paired_dataset(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare paired Chinese character dataset for simplified and traditional variants
    """
    df = df_full.copy()

    # Create one row per (simplified, traditional) pair
    df = df.query("`kSimplifiedVariant`.notna() or `kTraditionalVariant`.notna()").reset_index(drop=True)
    df["Unicode (S)"] = df["kSimplifiedVariant"].fillna(df["Unicode"]).str.split(" ")
    df["Unicode (T)"] = df["kTraditionalVariant"].fillna(df["Unicode"]).str.split(" ")
    df = df.explode("Unicode (S)", ignore_index=True)
    df = df.explode("Unicode (T)", ignore_index=True)

    # Remove rows where the simplified and traditional characters are the same
    df = df.query("`Unicode (S)` != `Unicode (T)`").reset_index(drop=True)

    df["Unicode Int (S)"] = df["Unicode (S)"].map(lambda val: int(str(val).removeprefix("U+"), 16))
    df["Unicode Int (T)"] = df["Unicode (T)"].map(lambda val: int(str(val).removeprefix("U+"), 16))
    df["Character (S)"] = df["Unicode Int (S)"].map(chr)
    df["Character (T)"] = df["Unicode Int (T)"].map(chr)

    df["Filename"] = df.groupby("Unicode")["Unicode Int"].transform(lambda vals: [f"{v}_{ix}.png" for ix, v in enumerate(vals)])
    df = df.set_index(sorted(c for c in df.columns if not c.startswith("k"))).reset_index()

    return df


def clean_definition(definition: str):
    """
    Clean a definition string by removing unwanted characters and patterns
    """
    if len(definition) == 0:
        return ""
    if any(s in definition for s in ["(Cant.", "Cantonese variant", "(J", "Japanese variant"]):
        return ""

    symbols = r'!\"#$%&\'()*+,-./:;<=>?@[\\'
    definition = re.sub(r"\(.*?[/u3400-\u9FFF]+.*?\)", "", definition)       # Remove any parentheses containing a Chinese character
    definition = re.sub(r",.*?[\u3400-\u9FFF]+.*?(,|$)", "", definition)     # Remove clauses containing a Chinese character separated by commas
    definition = re.sub(r";.*?[\u3400-\u9FFF]+.*?(;|$)", "", definition)     # Remove clauses containing a Chinese character separated by semicolons
    definition = re.sub(r"[\u3400-\u9FFF]", "", definition)                  # Remove all remaining Chinese characters and non-standard characters
    definition = re.sub(r"U\+\w+", "", definition)                           # Remove Unicode codes
    definition = re.sub(r"[^\x00-\x7F]", "", definition)                     # Keep only ASCII characters
    definition = re.sub(r'^[' + symbols + r']^_`{|}~]+', "", definition)     # Remove specific set of punctuation at start of string
    definition = re.sub(r'[' + symbols + r']^_`{|}~]+$', "", definition)     # Remove specific set of punctuation at end of string
    definition = re.sub(r'\s[' + symbols + r']^_`{|}~]+\s', " ", definition) # Remove isolated punctuation (surrounded by spaces)
    definition = re.sub(r"\s+", " ", definition).strip()                     # Trim extra spaces

    return definition


def create_image(character: str, font: ImageFont.FreeTypeFont, image_size: Tuple[int, int]):
    """
    Create an image of a single Unicode character
    """
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)
    img_w, img_h = image_size
    draw.text((img_w / 2, img_h * (15/32)), character, fill="black", font=font, anchor="mm")
    return image


def is_valid_char(character: str, font: ImageFont.FreeTypeFont, image_size: Tuple[int, int]):
    """
    Check if the character is supported by the font
    """
    image_char = create_image(character, font, image_size)
    image_unknown = create_image("�", font, image_size) # U+FFFD (invalid character)
    return image_char.tobytes() != image_unknown.tobytes()
