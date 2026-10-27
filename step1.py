import torch
from transformers import AutoModelForVision2Seq, BitsAndBytesConfig, AutoModelForVision2Seq

def load_model(mode):
    if mode == "int4":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif mode == "int8":
        quantization = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization = None
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        quantization_config=quantization,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map={"":0}
    )
    return model
mode = "int4"  # Change this to "int8" or "float16" as needed
model = load_model(mode)

print(f"Model loaded in {mode} mode, weights on GPU = {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")

