# Kaggle cell: load an image model next to the chat model. Run it any time; the running server
# picks up `pipe` immediately (no restart, same URL + token).
IMAGE_MODEL_ID = 'stabilityai/sdxl-turbo'
# Options for Kaggle T4s:
#   'stabilityai/sdxl-turbo'   ~7 GB, 512x512 in ~1-4 steps, good quality (non-commercial licence)
#   'stabilityai/sd-turbo'     ~3 GB, faster, lower quality (non-commercial licence)
# Non-turbo models (e.g. 'stabilityai/stable-diffusion-xl-base-1.0') need more steps:
#   set COGNIVO_IMAGE_STEPS=30 and COGNIVO_IMAGE_GUIDANCE=7 below.
%pip install -q diffusers accelerate
import os, gc, torch
from diffusers import AutoPipelineForText2Image
os.environ['COGNIVO_IMAGE_STEPS'] = '4'
os.environ['COGNIVO_IMAGE_GUIDANCE'] = '0'
globals().pop('pipe', None); gc.collect(); torch.cuda.empty_cache()
# Put the image model on the last GPU so it doesn't fight the chat model (T4 x2 -> cuda:1).
device = f'cuda:{torch.cuda.device_count() - 1}'
pipe = AutoPipelineForText2Image.from_pretrained(IMAGE_MODEL_ID, torch_dtype=torch.float16, variant='fp16').to(device)
pipe.set_progress_bar_config(disable=True)
print('Image model', IMAGE_MODEL_ID, 'loaded on', device)
