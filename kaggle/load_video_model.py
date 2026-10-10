# Kaggle cell: load a short-video model next to chat/image/vision models.
# Run any time after the server is up - it is picked up immediately (same URL + token).
# Videos take minutes on a T4, so the website runs them as background jobs and polls for progress.
VIDEO_MODEL_ID = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'  # ~7 GB; the only Kaggle-T4-friendly open video model
%pip install -q "diffusers>=0.32" accelerate ftfy imageio imageio-ffmpeg
import gc, torch
from diffusers import WanPipeline, AutoencoderKLWan

globals().pop('video_pipe', None); gc.collect(); torch.cuda.empty_cache()
device = f'cuda:{torch.cuda.device_count() - 1}'  # last GPU; keep the chat model on cuda:0

vae = AutoencoderKLWan.from_pretrained(VIDEO_MODEL_ID, subfolder='vae', dtype=torch.float32)
video_pipe = WanPipeline.from_pretrained(VIDEO_MODEL_ID, vae=vae, torch_dtype=torch.bfloat16).to(device)
video_pipe.set_progress_bar_config(disable=True)
import os
os.environ['COGNIVO_VIDEO_STEPS'] = '30'   # lower (e.g. 20) for faster/rougher clips
print('Video model', VIDEO_MODEL_ID, 'loaded on', device)
print('NOTE: a 3s clip at 30 steps takes roughly 3-8 minutes on a T4 - this is expected.')
