# 生成字 (ShēngChéngZì)

| EN | Generative AI for Chinese Characters |
| --- | --- |
| ZH | 汉字的生成式人工智能 |

---

## Overview

生成字 (ShēngChéngZì) is a Generative AI toolkit for various tasks involving Chinese characters, including:

- **Random-to-Character (R2C)**  
For a given random latent vector, generate an image of a Chinese character.

- **Text-to-Character (T2C)**  
For a given text prompt, generate an image of a Chinese character.

- **Character-to-Character (C2C)**  
For a given Chinese character, generate an image of its simplified or traditional variant.

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

3. Generate the dataset(s) of Chinese characters using the notebooks in [`core/dataset`](core/dataset)
4. Train models using the Python scripts in the repository root (see below)
5. (WIP) Run inference on the models

---

## Training

The repository root contains Python scripts for training different models on various tasks.

### Available Models

- **Glyffuser**
  - Random-to-Character: `python train_glyffuser_r2c.py`
  - Text-to-Character: `python train_glyffuser_t2c.py`
- **Palette**
  - Character-to-Character: `python train_palette_c2c.py`
- **ShengChengZi**
  - Character-to-Character: `python train_scz_c2c.py`
  - Character-to-Character (New): `python train_scz_c2c_new.py`

All models output TensorBoard logs, checkpoints, and evaluation images to their respective output directories in `out/`.

---

## Acknowledgements

The `glyffuser` model and code is based upon the [article](https://yue-here.com/posts/glyffuser/) and [repo](https://github.com/yue-here/glyffuser/tree/main) by Yue Wu.

The `palette` model and code is based upon the [paper](https://doi.org/10.1145/3528233.353075) and [repo](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models) by Chitwan Saharia et al.
