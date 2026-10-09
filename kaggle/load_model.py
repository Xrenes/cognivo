# Kaggle cell 1: load the model. To switch models later: change MODEL_ID and rerun ONLY this cell.
# The server and tunnel keep running (same URL + token); in the website click Settings -> Test connection.
# Settings: Accelerator = GPU T4 x2, Internet = On.
MODEL_ID = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
# Options that fit Kaggle T4s (16 GB each):
#   'Qwen/Qwen2.5-Coder-1.5B-Instruct'   fast, ~3 GB
#   'Qwen/Qwen2.5-Coder-3B-Instruct'     better, ~7 GB
#   'Qwen/Qwen2.5-Coder-7B-Instruct'     best, ~15 GB, split over both T4s
#   'Qwen/Qwen2.5-7B-Instruct'           general chat instead of coding
import os, torch, gc
os.environ['HF_HOME'] = '/tmp/hf_cache'
os.environ['COGNIVO_MODEL_ID'] = MODEL_ID  # the running server picks this up immediately
from transformers import AutoModelForCausalLM, AutoTokenizer
# free a previously loaded model when switching
for _n in ('model', 'tokenizer'):
    globals().pop(_n, None)
gc.collect(); torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16, device_map='auto')
model.eval()
print('Loaded', MODEL_ID, 'on', model.hf_device_map if hasattr(model, 'hf_device_map') else model.device)
