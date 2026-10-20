"""Merge the QLoRA adapter into a full fp16 model so it can be converted to ONNX for transformers.js.

  python merge_export.py --adapter out/adapter --out out/merged                     # local only
  python merge_export.py --adapter out/adapter --out out/merged --push yourname/trainer-coach-smollm2-360m

Merge into the fp16 base (NOT the 4-bit one): merging into quantised weights loses precision.
Then convert to ONNX (see README "Ship it to the browser").
"""
import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", default="out/merged")
ap.add_argument("--push", help="HF repo id to upload the merged model to (needs HF_TOKEN)")
a = ap.parse_args()

base = json.load(open(os.path.join(a.adapter, "train_info.json")))["base"]
print("base:", base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
tok = AutoTokenizer.from_pretrained(a.adapter)
model.save_pretrained(a.out, safe_serialization=True)
tok.save_pretrained(a.out)
print("saved merged model to", a.out)
if a.push:
    token = os.environ["HF_TOKEN"]
    model.push_to_hub(a.push, token=token)
    tok.push_to_hub(a.push, token=token)
    print("pushed to https://huggingface.co/" + a.push)
