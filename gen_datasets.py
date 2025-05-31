#!/usr/bin/env python3
"""
Generates image datasets of Chinese characters from Unihan data and font files
"""

from pathlib import Path

import click

from core.utils.dataset_utils import gen_dataset, load_unihan_data  # NOQA
from core.utils.repo_utils import get_repo_dir


@click.command()
@click.option("-d", "--dataset-type", type=str, default="both", help="Type of dataset to generate ('unpaired', 'paired', or 'both')")
@click.option("-i", "--image-size",   type=int, default=64,     help="Image size in pixels")
@click.option("-f", "--font-size",    type=int, default=None,     help="Font size of characters in images (None to auto-scale)")
def main(dataset_type: str, image_size: int, font_size: int):
    """ Generate Chinese character datasets from Unihan data and font files. """
    dataset_type = dataset_type.lower()
    image_size = (image_size, image_size)
    font_size = font_size if font_size is not None else int(100 * (image_size[0] / 128))

    base_data_dir = Path(get_repo_dir()) / "data"
    unihan_dir = base_data_dir / "unihan"
    out_dir = base_data_dir / "datasets"
    font_dir = base_data_dir / "fonts"

    df_full = load_unihan_data(unihan_dir)

    dataset_types = ["unpaired", "paired"] if dataset_type == "both" else [dataset_type]
    for data_type in dataset_types:
        out_dir_data = out_dir / f"{data_type}_{image_size[0]}x{image_size[1]}"
        gen_dataset(df_full, data_type, out_dir_data, font_dir, image_size, font_size)


if __name__ == "__main__":
    main()
