# Kaggle cell (optional): a Gradio UI inside Kaggle for testing without the website.
# Run AFTER load_model.py and load_image_model.py. Works alongside the website server
# (shares the same models and GPU lock). Prints a public *.gradio.live link + login.
%pip install -q gradio
import os, math, secrets, threading, torch, gradio as gr
from PIL import Image
from diffusers import AutoPipelineForImage2Image

# Share the website server's GPU lock if it is running, so jobs never overlap on the GPU.
try:
    from kaggle_server import lock as gpu_lock
except Exception:
    gpu_lock = threading.Lock()

STEPS = int(os.environ.get('COGNIVO_IMAGE_STEPS', '4'))
GUIDANCE = float(os.environ.get('COGNIVO_IMAGE_GUIDANCE', '0'))
_i2i = {}
SIZES = (512, 768, 1024)
def _size(v):
    v = int(v); return v if v in SIZES else 512
def _strength(v):
    return min(1.0, max(0.2, float(v)))
def _text(t, n=1000):
    return (t or '').strip()[:n]

def _img2img():
    if 'pipe' not in globals(): raise gr.Error('No image model loaded - run load_image_model.py first')
    if _i2i.get('for') is not pipe:
        _i2i['pipe'], _i2i['for'] = AutoPipelineForImage2Image.from_pipe(pipe), pipe
    return _i2i['pipe']

def _gpu(fn):
    if not gpu_lock.acquire(timeout=120): raise gr.Error('GPU busy - try again')
    try:
        with torch.inference_mode(): return fn()
    finally: gpu_lock.release()

def edit_photo(photo, prompt, strength, size):
    if photo is None: raise gr.Error('Upload a photo first')
    prompt, size, strength = _text(prompt), _size(size), _strength(strength)
    if not prompt: raise gr.Error('Describe the change')
    src = photo.convert('RGB').resize((size, size))
    steps = max(STEPS, math.ceil(1 / strength))  # turbo models need steps*strength >= 1
    return _gpu(lambda: _img2img()(prompt=prompt, image=src, strength=strength,
                                    num_inference_steps=steps, guidance_scale=GUIDANCE).images[0])

def generate(prompt, size):
    if 'pipe' not in globals(): raise gr.Error('No image model loaded - run load_image_model.py first')
    prompt, size = _text(prompt), _size(size)
    if not prompt: raise gr.Error('Describe the image')
    return _gpu(lambda: pipe(prompt=prompt, width=size, height=size,
                             num_inference_steps=STEPS, guidance_scale=GUIDANCE).images[0])

def chat(message, history):
    if 'model' not in globals(): raise gr.Error('No chat model loaded - run load_model.py first')
    msgs = [{'role': 'system', 'content': 'You are Cognivo, a helpful AI assistant.'}]
    msgs += [{'role': m['role'], 'content': _text(m['content'], 12000)} for m in history[-14:]
             if m.get('role') in ('user', 'assistant') and isinstance(m.get('content'), str)]
    msgs.append({'role': 'user', 'content': _text(message, 12000)})
    def run():
        enc = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt', return_dict=True).to(model.device)
        out = model.generate(**enc, max_new_tokens=1024, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][enc['input_ids'].shape[-1]:], skip_special_tokens=True)
    return _gpu(run)

with gr.Blocks(title='Cognivo (Kaggle)', theme=gr.themes.Soft(primary_hue='orange')) as demo:
    gr.Markdown('## Cognivo - Kaggle')
    with gr.Tab('Edit Photo'):
        with gr.Row():
            with gr.Column():
                e_in = gr.Image(type='pil', label='Upload photo', sources=['upload', 'clipboard'])
                e_prompt = gr.Textbox(label='What should change?', placeholder='make it a watercolor painting')
                e_strength = gr.Slider(0.2, 1.0, value=0.6, step=0.05, label='Change strength (low = keep more of the photo)')
                e_size = gr.Radio([512, 768, 1024], value=512, label='Size')
                e_btn = gr.Button('Generate', variant='primary')
            e_out = gr.Image(label='Result', format='png')
        e_btn.click(edit_photo, [e_in, e_prompt, e_strength, e_size], e_out)
    with gr.Tab('Generate Image'):
        with gr.Row():
            with gr.Column():
                g_prompt = gr.Textbox(label='Describe the image', placeholder='a fox in a neon city, cinematic')
                g_size = gr.Radio([512, 768, 1024], value=512, label='Size')
                g_btn = gr.Button('Generate', variant='primary')
            g_out = gr.Image(label='Result', format='png')
        g_btn.click(generate, [g_prompt, g_size], g_out)
    with gr.Tab('Chat'):
        gr.ChatInterface(chat, type='messages')

# Public link is protected by a login; share it only with people you trust.
GRADIO_PASSWORD = secrets.token_urlsafe(9)
demo.queue(default_concurrency_limit=1)
demo.launch(share=True, auth=('cognivo', GRADIO_PASSWORD), prevent_thread_lock=True, show_error=True)
print('=' * 64)
print(' Gradio login  ->  user: cognivo   password:', GRADIO_PASSWORD)
print(' Open the *.gradio.live link printed above. Rerun to restart with a new password')
print(' (call demo.close() first if a previous Gradio UI is still running).')
print('=' * 64)
