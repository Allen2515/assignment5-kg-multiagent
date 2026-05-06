"""
llm_loader.py (Assignment 5)
----------------------------
Thin wrapper around the A4 HuggingFace model cache.

Resolution order for cache directory:
  1. Assignment-4-main/hf_model_cache/ next to this file (A4 already downloaded)
  2. hf_model_cache/ next to this file (fresh download)
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import Any

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

_this_dir = os.path.dirname(os.path.abspath(__file__))
_a4_cache = os.path.join(_this_dir, "Assignment-4-main", "hf_model_cache")
_a5_cache = os.path.join(_this_dir, "hf_model_cache")
MODEL_CACHE_DIR = _a4_cache if os.path.exists(_a4_cache) else _a5_cache

_llm_instance = None
_tokenizer = None
_raw_pipeline = None


def load_local_llm(model_id: str = MODEL_ID) -> Any:
    global _llm_instance, _tokenizer, _raw_pipeline
    if _llm_instance is not None:
        return _llm_instance

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    cache_slug = "models--" + model_id.replace("/", "--")
    if os.path.exists(os.path.join(MODEL_CACHE_DIR, cache_slug)):
        print(f"[llm_loader] Loading '{model_id}' from local cache: {MODEL_CACHE_DIR}")
    else:
        print(f"[llm_loader] First run: downloading '{model_id}' to {MODEL_CACHE_DIR} ...")

    _tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=MODEL_CACHE_DIR)

    model_kwargs: dict[str, Any] = {"cache_dir": MODEL_CACHE_DIR, "torch_dtype": dtype}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    _raw_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=_tokenizer,
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.1,
        return_full_text=False,
    )

    _llm_instance = _raw_pipeline
    print(f"[llm_loader] Model ready on {device.upper()}.\n")
    return _llm_instance


def get_tokenizer():
    return _tokenizer


def get_raw_pipeline():
    return _raw_pipeline
