# 生成字 (ShēngChéng Zì)

| EN | Generative AI for Chinese Characters |
| --- | --- |
| ZH | 汉字的生成式人工智能 |

---

## Overview

生成字 (ShēngChéng Zì) is a Generative AI toolkit for various tasks involving Chinese characters, including:

- **Text-to-Image Generation**
  - Generate a Chinese character based on a text prompt (see `glyffuser`'s rand2char/text2char models)
- **Image-to-Image Translation**
  - Generate a simplified or traditional variant of a given Chinese character (see `palette`'s char2char models)

---

## Getting Started

1. Clone the repository

    ```bash
    git clone https://github.com/awisniewski21/ShengChengZi.git
    ```

2. Install dependencies

    ```bash
    pip install -r requirements.txt
    ```

3. Generate the dataset(s) of Chinese characters using [`notebooks/generate_datasets.ipynb`](notebooks/generate_datasets.ipynb)
4. Train models using the Python runner scripts in the `runners/` directory
5. (WIP) Run inference on the model(s)

---

## Training

The `runners/` directory contains Python scripts for training different models, converted from Jupyter notebooks for streamlined execution. All scripts should be run from the repository root:

### Available Models
- **Palette (char2char)**: `python runners/run_palette.py -c palette/config/char2char.json -p train`
- **Character-to-Character**: `python runners/run_char2char.py`
- **Character-to-Character (New)**: `python runners/run_char2char_new.py`
- **Random-to-Character**: `python runners/run_rand2char.py`
- **Text-to-Character**: `python runners/run_text2char.py`

All models output TensorBoard logs, checkpoints, and evaluation images to their respective output directories.

---

## Acknowledgements

Portions of the source code are based upon the [glyffuser repo](https://github.com/yue-here/glyffuser/tree/main) and [corresponding article](https://yue-here.com/posts/glyffuser/).
