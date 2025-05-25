import os
import re
from typing import Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def clean_definition(definition: str):
    if len(definition) == 0: return ""
    if any(s in definition for s in [
        "(Cant.", "Cantonese variant",
        "(J", "Japanese variant",
    ]):
        return ""

    # Remove any parentheses containing a Chinese character
    definition = re.sub(r"\(.*?[/u3400-\u9FFF]+.*?\)", "", definition)

    # Remove clauses containing a Chinese character separated by commas or semicolons
    definition = re.sub(r",.*?[\u3400-\u9FFF]+.*?(,|$)", "", definition)
    definition = re.sub(r";.*?[\u3400-\u9FFF]+.*?(;|$)", "", definition)

    # Remove all remaining Chinese characters and non-standard characters
    definition = re.sub(r"[\u3400-\u9FFF]", "", definition)

    # Remove Unicode codes
    definition = re.sub(r"U\+\w+", "", definition)

    # Keep only ASCII
    definition = re.sub(r"[^\x00-\x7F]", "", definition)

    # Remove specific set of punctuation at start or end of string
    definition = re.sub(r'^[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+|[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+$', "", definition)

    # Remove isolated punctuation (surrounded by spaces)
    definition = re.sub(r'\s[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+\s', " ", definition)

    # Trim extra spaces
    definition = re.sub(r"\s+", " ", definition).strip()

    return definition


def generate_and_save_character_images(
    df: pd.DataFrame,
    font_path: str,
    out_dir: str,
    image_size: Tuple[int, int] = (128, 128),
    font_size: int = 100,
):
    """
    Generate and save images for each character specified in the DataFrame's column.
    """
    font = ImageFont.truetype(font_path, font_size)
    os.makedirs(out_dir, exist_ok=True)

    invalid_ixs = []
    for ix, row in tqdm(df.iterrows(), total=len(df)):
        filename = os.path.join(out_dir, row["Filename"])
        if os.path.exists(filename):
            continue

        if not is_valid_char(row["Character"], font, image_size=image_size):
            invalid_ixs.append(ix)
            continue

        img = create_image(row["Character"], font, image_size=image_size)
        img.save(filename)

    return df.drop(index=invalid_ixs)


def create_image(character, font, image_size=(128, 128)):
    """
    Create an image of a single Unicode character
    """
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)
    text_width, text_height = draw.textbbox((0, 0), character, font=font)[2:]
    y_offset = 16 * (image_size[1] / 128)
    x = (image_size[0] - text_width) / 2
    y = (image_size[1] - text_height) / 2 - y_offset
    draw.text((x, y), character, fill="black", font=font)
    return image


def is_valid_char(character, font, image_size=(128, 128)):
    """
    Check if the character is supported by the font
    """
    image_char = create_image(character, font, image_size)
    image_unknown = create_image("�", font, image_size)  # U+FFFD is the replacement character
    return image_char.tobytes() != image_unknown.tobytes()
