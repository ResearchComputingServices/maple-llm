import argparse
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from pathlib import Path
import torch
import timeit

available_devices = ['cpu']
if torch.cuda.is_available():
    device = 'cuda'
    available_devices.append('cuda')
else:
    device = 'cpu'
    
argparser = argparse.ArgumentParser()
argparser.add_argument("-m","--mistral_folder", type=str, default=None, help="Destination folder for downloading Mistral models")

load_dotenv('production.env')

args = argparser.parse_args()
if args.mistral_folder is not None:
    mistral_models_path = Path(args.mistral_folder).joinpath('mistral_models', '7B-v0.3')
else:
    mistral_models_path = Path.home().joinpath('mistral_models', '7B-v0.3')
    
mistral_models_path.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="mistralai/Mistral-7B-v0.3", 
    allow_patterns=[
        "params.json", 
        "consolidated.safetensors", 
        "tokenizer.model.v3"], 
    local_dir=mistral_models_path
    )

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)
inputs = tokenizer("Hello my name is", return_tensors="pt")
for dev in available_devices:
    print(f"Device: {dev}")
    model.to(dev)
    inputs.to(dev)

    tstart  = timeit.default_timer()
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    tend = timeit.default_timer()
    print(f"Time: {tend-tstart} s")






