# Kaggle cell: load a vision model so the website's chat can SEE photos you attach.
# Run any time after the server is up - it is picked up immediately (same URL + token).
VISION_MODEL_ID = 'Qwen/Qwen2-VL-2B-Instruct'   # ~4.5 GB in fp16; fits next to SDXL-Turbo on the 2nd T4
# Bigger/better option (~8 GB, may not fit next to the image model): 'Qwen/Qwen2.5-VL-3B-Instruct'
%pip install -q -U "transformers>=4.49" accelerate
import gc, torch
from transformers import AutoProcessor, AutoModelForImageTextToText
globals().pop('vlm', None); globals().pop('vlm_processor', None); gc.collect(); torch.cuda.empty_cache()
device = f'cuda:{torch.cuda.device_count() - 1}'  # last GPU; the chat model stays on cuda:0
# Limit pixels per image so big photos don't run the T4 out of memory.
vlm_processor = AutoProcessor.from_pretrained(VISION_MODEL_ID, min_pixels=256*28*28, max_pixels=768*28*28)
vlm = AutoModelForImageTextToText.from_pretrained(VISION_MODEL_ID, dtype=torch.float16).to(device).eval()
print('Vision model', VISION_MODEL_ID, 'loaded on', device)
