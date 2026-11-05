"""
step1.py - Load OpenVLA-7B at a chosen precision and print how much GPU memory the weights use.

Purpose: check that the compressed model fits the 16 GB budget (ECE 535 project).

Setup (on a machine with an NVIDIA GPU):
    nvidia-smi                                   # check the GPU and its memory
    git clone https://github.com/izawabno/ECE535Project
    cd ECE535Project
    python -m venv vla-env
    source vla-env/bin/activate                  # Windows: vla-env\\Scripts\\activate
    pip install torch
    pip install transformers==4.46.3 tokenizers==0.20.3 timm==0.9.16 accelerate bitsandbytes pillow

Run (one mode per run):
    python step1.py int4
    python step1.py int8
    python step1.py fp16

Notes:
    - The first run downloads about 15 GB from Hugging Face.
    - Measured on a Colab T4: int4 = 4.36 GB, int8 = 7.40 GB. fp16 is about 15 GB or more.
    - Close Isaac Sim first so the number is clean.
    - A warning about transformers==4.40.1 is expected and harmless here.
"""
import sys

import torch
from transformers import AutoModelForVision2Seq, BitsAndBytesConfig


def load_model(mode):
    if mode == "int4":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif mode == "int8":
        quantization = BitsAndBytesConfig(load_in_8bit=True)
    elif mode == "fp16":
        quantization = None  # no compression: full-size baseline
    else:
        raise SystemExit("Mode must be int4, int8, or fp16.")

    return AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        quantization_config=quantization,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map={"": 0},
    )


if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU found. Run this on a machine with an NVIDIA GPU.")

mode = sys.argv[1] if len(sys.argv) > 1 else "int4"
model = load_model(mode)

print(f"Model loaded in {mode} mode, weights on GPU = {torch.cuda.memory_allocated() / 1024**3:.2f} GB")