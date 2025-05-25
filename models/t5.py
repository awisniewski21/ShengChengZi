from typing import List, Tuple

import torch
import transformers
from einops import rearrange
from transformers import T5Config, T5EncoderModel, T5Tokenizer

T5_CONFIGS = {}
MAX_LENGTH = 256
DEFAULT_T5_NAME = "google/t5-v1_1-base"

transformers.logging.set_verbosity_error()

def get_tokenizer(name) -> T5Tokenizer:
    return T5Tokenizer.from_pretrained(name, model_max_length=MAX_LENGTH)

def get_model(name):
    return T5EncoderModel.from_pretrained(name)

def get_model_and_tokenizer(name) -> Tuple[T5EncoderModel, T5Tokenizer]:
    global T5_CONFIGS

    if name not in T5_CONFIGS:
        T5_CONFIGS[name] = dict()
    if "model" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["model"] = get_model(name)
    if "tokenizer" not in T5_CONFIGS[name]:
        T5_CONFIGS[name]["tokenizer"] = get_tokenizer(name)

    return T5_CONFIGS[name]["model"], T5_CONFIGS[name]["tokenizer"]

def get_encoded_dim(name):
    if name not in T5_CONFIGS:
        # avoids loading the model if we only want to get the dim
        config = T5Config.from_pretrained(name)
        T5_CONFIGS[name] = dict(config=config)
    elif "config" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["config"]
    elif "model" in T5_CONFIGS[name]:
        config = T5_CONFIGS[name]["model"].config
    else:
        assert False
    return config.d_model

# encoding text

def t5_tokenize(texts: List[str], name: str = DEFAULT_T5_NAME, device: str | None = None):
    t5, tokenizer = get_model_and_tokenizer(name)

    if device is not None:
        t5 = t5.to(device)
    elif torch.cuda.is_available():
        t5 = t5.cuda()

    device = next(t5.parameters()).device

    encoded = tokenizer.batch_encode_plus(
        texts,
        return_tensors = "pt",
        padding = "longest",
        max_length = MAX_LENGTH,
        truncation = True
    )

    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)
    return input_ids, attn_mask


def t5_encode_tokenized_text(
    token_ids,
    attn_mask = None,
    pad_id = None,
    name = DEFAULT_T5_NAME,
    device: str | None = None,
):
    assert attn_mask is not None or pad_id is not None
    t5, _ = get_model_and_tokenizer(name)

    if device is not None:
        t5 = t5.to(device)
    elif torch.cuda.is_available():
        t5 = t5.cuda()

    attn_mask = attn_mask if attn_mask is not None else (token_ids != pad_id).long()

    t5.eval()
    with torch.no_grad():
        output = t5(input_ids=token_ids, attention_mask=attn_mask)
        encoded_text = output.last_hidden_state.detach()

    attn_mask = attn_mask.bool()

    return encoded_text.masked_fill(~rearrange(attn_mask, "... -> ... 1"), 0.) # just force all embeddings that is padding to be equal to 0.


def t5_encode_text(texts: List[str], name: str = DEFAULT_T5_NAME, return_attn_mask: bool = False, device: str | None = None):
    token_ids, attn_mask = t5_tokenize(texts, name=name, device=device)
    encoded_text = t5_encode_tokenized_text(token_ids, attn_mask=attn_mask, name=name, device=device)

    if return_attn_mask:
        attn_mask = attn_mask.bool()
        return encoded_text, attn_mask

    return encoded_text
