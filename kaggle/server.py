"""Use in Kaggle after loading global `model` and `tokenizer` objects."""
import os, time, secrets, threading
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Annotated, Literal
import torch

def model_id():  # read per request so the notebook can switch models without restarting the server
    return os.environ.get('COGNIVO_MODEL_ID', 'Qwen/Qwen2.5-Coder-1.5B-Instruct')
API_KEY = os.environ.get('COGNIVO_API_KEY')
if not API_KEY or len(API_KEY) < 24:
    raise RuntimeError('Set COGNIVO_API_KEY to a random 24+ character secret')
ALLOWED_ORIGINS = [x.strip().rstrip('/') for x in os.environ.get('COGNIVO_ALLOWED_ORIGINS', '').split(',') if x.strip()]
if not ALLOWED_ORIGINS:
    raise RuntimeError('Set COGNIVO_ALLOWED_ORIGINS to your exact website origin')
app = FastAPI(title='Cognivo Kaggle inference API',docs_url=None,redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                   allow_methods=['GET','POST','OPTIONS'], allow_headers=['Authorization','Content-Type'])
lock = threading.Lock()

class ImageURL(BaseModel):
    url: str = Field(max_length=4_000_000)  # data:image/...;base64,...
class Part(BaseModel):
    type: Literal['text','image_url']
    text: str|None = Field(default=None, max_length=12000)
    image_url: ImageURL|None = None
class Message(BaseModel):
    role: Literal['system','user','assistant']
    # plain text, or at most 6 parts (text + up to 4 images); 16 messages max per request
    content: Annotated[str, Field(min_length=1, max_length=12000)] | Annotated[list[Part], Field(min_length=1, max_length=6)]

    def text(self):
        if isinstance(self.content, str): return self.content
        return '\n'.join(p.text for p in self.content if p.type=='text' and p.text)[:12000]
    def images(self):
        return [] if isinstance(self.content, str) else [p.image_url.url for p in self.content if p.type=='image_url' and p.image_url]

MAX_IMAGES = 4
MAX_PIXELS = 4096 * 4096  # reject decompression bombs before any pixel data is decoded
def open_image(b64):
    import io, base64
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64, validate=True)))  # header only, lazy
        if img.width * img.height > MAX_PIXELS: raise ValueError('too many pixels')
        img.thumbnail((1024, 1024))  # cap size/memory regardless of what the client sent
        return img.convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail='Invalid or too large image (max 4096x4096)')
def decode_image(url):
    if not url.startswith('data:image/') or ';base64,' not in url:
        raise HTTPException(status_code=400, detail='Images must be base64 data URLs')
    return open_image(url.split(',',1)[1])
class Completion(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1,max_length=16)
    max_tokens: int = Field(default=512,ge=1,le=1024)
    temperature: float = Field(default=0.3,ge=0,le=2)

def authorize(header):
    prefix='Bearer '
    if not header or not header.startswith(prefix) or not secrets.compare_digest(header[len(prefix):],API_KEY):
        raise HTTPException(status_code=401,detail='Invalid access token')

class ImageRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    size: Literal['512x512','768x768','1024x1024'] = '512x512'
    n: int = Field(default=1, ge=1, le=1)
    model: str | None = None
    response_format: str | None = None

@app.post('/v1/images/generations')
def images(req: ImageRequest, authorization: str|None=Header(default=None)):
    authorize(authorization)
    import __main__, io, base64
    pipe = getattr(__main__, 'pipe', None)
    if pipe is None: raise HTTPException(status_code=503, detail='No image model loaded (run load_image_model.py)')
    if not lock.acquire(blocking=False): raise HTTPException(status_code=429, detail='GPU busy; retry shortly')
    try:
        w, h = (int(x) for x in req.size.split('x'))
        steps = int(os.environ.get('COGNIVO_IMAGE_STEPS', '4'))
        guidance = float(os.environ.get('COGNIVO_IMAGE_GUIDANCE', '0'))
        with torch.inference_mode():
            img = pipe(prompt=req.prompt, width=w, height=h, num_inference_steps=steps, guidance_scale=guidance).images[0]
        buf = io.BytesIO(); img.save(buf, format='PNG')
        return {'created': int(time.time()), 'data': [{'b64_json': base64.b64encode(buf.getvalue()).decode()}]}
    finally: lock.release()

class ImageEditRequest(ImageRequest):
    image: str = Field(min_length=100, max_length=4_000_000)  # base64 JPEG/PNG from the website
    strength: float = Field(default=0.6, ge=0.1, le=1.0)      # how much to change the photo

@app.post('/v1/images/edits')
def image_edit(req: ImageEditRequest, authorization: str|None=Header(default=None)):
    authorize(authorization)
    import __main__, io, base64, math
    from PIL import Image
    pipe = getattr(__main__, 'pipe', None)
    if pipe is None: raise HTTPException(status_code=503, detail='No image model loaded (run load_image_model.py)')
    src = open_image(req.image)
    if not lock.acquire(blocking=False): raise HTTPException(status_code=429, detail='GPU busy; retry shortly')
    try:
        from diffusers import AutoPipelineForImage2Image
        # reuse the loaded weights; rebuild only when the notebook loads a different image model
        if getattr(__main__, '_cognivo_i2i_for', None) is not pipe:
            __main__._cognivo_i2i = AutoPipelineForImage2Image.from_pipe(pipe)
            __main__._cognivo_i2i_for = pipe
        w, h = (int(x) for x in req.size.split('x'))
        src = src.resize((w, h))
        steps = int(os.environ.get('COGNIVO_IMAGE_STEPS', '4'))
        steps = max(steps, math.ceil(1 / req.strength))  # turbo models need steps*strength >= 1
        guidance = float(os.environ.get('COGNIVO_IMAGE_GUIDANCE', '0'))
        with torch.inference_mode():
            img = __main__._cognivo_i2i(prompt=req.prompt, image=src, strength=req.strength,
                                        num_inference_steps=steps, guidance_scale=guidance).images[0]
        buf = io.BytesIO(); img.save(buf, format='PNG')
        return {'created': int(time.time()), 'data': [{'b64_json': base64.b64encode(buf.getvalue()).decode()}]}
    finally: lock.release()

@app.get('/v1/models')
def models(authorization: str|None=Header(default=None)):
    authorize(authorization)
    return {'object':'list','data':[{'id':model_id(),'object':'model','owned_by':'self'}]}

def vision_reply(messages, kwargs):
    """Messages containing photos go to the vision model (`vlm` + `vlm_processor` from load_vision_model.py)."""
    import __main__
    vlm, proc = getattr(__main__,'vlm',None), getattr(__main__,'vlm_processor',None)
    if vlm is None or proc is None:
        raise HTTPException(status_code=400, detail='No vision model loaded - run load_vision_model.py in Kaggle')
    urls = [u for m in messages for u in m.images()]
    if len(urls) > MAX_IMAGES: raise HTTPException(status_code=400, detail=f'At most {MAX_IMAGES} images per request')
    images, vl_msgs = [], []
    for m in messages:
        parts = [{'type':'image'} for _ in m.images()]
        images += [decode_image(u) for u in m.images()]
        if m.text(): parts.append({'type':'text','text':m.text()})
        vl_msgs.append({'role':m.role,'content':parts})
    prompt = proc.apply_chat_template(vl_msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[prompt], images=images, return_tensors='pt').to(vlm.device)
    with torch.inference_mode(): out = vlm.generate(**inputs, **kwargs)
    return proc.batch_decode(out[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)[0]

class VideoRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    size: Literal['832x480','480x832'] = '832x480'
    num_frames: Literal[33,49,81] = 49   # ~2s/3s/5s at 16fps

VIDEO_DIR = '/kaggle/working/cognivo_videos'
os.makedirs(VIDEO_DIR, exist_ok=True)
jobs = {}            # id -> {status, progress, error, owner}
jobs_lock = threading.Lock()
MAX_QUEUED_PER_TOKEN = 2  # a client can't pile up unlimited background GPU work
MAX_SAVED_VIDEOS = 20     # keep Kaggle's disk from filling up over a long session

def _cleanup_old_videos():
    files = sorted((f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')),
                   key=lambda f: os.path.getmtime(os.path.join(VIDEO_DIR, f)))
    for f in files[:-MAX_SAVED_VIDEOS]:
        try: os.remove(os.path.join(VIDEO_DIR, f))
        except OSError: pass

def _run_video_job(job_id, prompt, w, h, num_frames):
    import __main__
    try:
        vid_pipe = getattr(__main__, 'video_pipe', None)
        if vid_pipe is None:
            raise RuntimeError('No video model loaded - run load_video_model.py in Kaggle')
        if not lock.acquire(timeout=1800):  # wait for any chat/image job ahead of it, up to 30 min
            raise RuntimeError('Timed out waiting for the GPU')
        try:
            from diffusers.utils import export_to_video
            steps = int(os.environ.get('COGNIVO_VIDEO_STEPS', '30'))
            def cb(pipe_, step, timestep, kw):
                with jobs_lock: jobs[job_id]['progress'] = min(0.99, (step + 1) / steps)
                return kw
            with torch.inference_mode():
                out = vid_pipe(prompt=prompt, height=h, width=w, num_frames=num_frames,
                               num_inference_steps=steps, callback_on_step_end=cb).frames[0]
            path = os.path.join(VIDEO_DIR, job_id + '.mp4')
            export_to_video(out, path, fps=16)
            with jobs_lock: jobs[job_id].update(status='done', progress=1.0, path=path)
            _cleanup_old_videos()
        finally:
            lock.release()
    except Exception as e:
        with jobs_lock: jobs[job_id].update(status='failed', error=str(e)[:300])

@app.post('/v1/videos')
def create_video(req: VideoRequest, authorization: str|None=Header(default=None)):
    authorize(authorization)
    import __main__
    if getattr(__main__, 'video_pipe', None) is None:
        raise HTTPException(status_code=503, detail='No video model loaded (run load_video_model.py)')
    with jobs_lock:
        active = sum(1 for j in jobs.values() if j['owner']==authorization and j['status'] in ('queued','running'))
        if active >= MAX_QUEUED_PER_TOKEN:
            raise HTTPException(status_code=429, detail='Too many videos already queued - wait for one to finish')
        job_id = secrets.token_hex(8)
        jobs[job_id] = {'status': 'queued', 'progress': 0.0, 'owner': authorization}
    w, h = (int(x) for x in req.size.split('x'))
    threading.Thread(target=_run_video_job, args=(job_id, req.prompt, w, h, req.num_frames), daemon=True).start()
    return {'id': job_id, 'status': 'queued'}

@app.get('/v1/videos/{job_id}')
def video_status(job_id: str, authorization: str|None=Header(default=None)):
    authorize(authorization)
    with jobs_lock: j = jobs.get(job_id)
    if not j or j['owner'] != authorization: raise HTTPException(status_code=404, detail='Unknown job')
    return {'id': job_id, 'status': j['status'], 'progress': j.get('progress', 0), 'error': j.get('error')}

@app.get('/v1/videos/{job_id}/content')
def video_content(job_id: str, authorization: str|None=Header(default=None)):
    authorize(authorization)
    with jobs_lock: j = jobs.get(job_id)
    if not j or j['owner'] != authorization: raise HTTPException(status_code=404, detail='Unknown job')
    if j['status'] != 'done': raise HTTPException(status_code=409, detail=f"Video is {j['status']}, not ready")
    return FileResponse(j['path'], media_type='video/mp4', filename='cognivo.mp4')

@app.post('/v1/chat/completions')
def complete(req:Completion,authorization:str|None=Header(default=None)):
    authorize(authorization)
    if req.model!=model_id(): raise HTTPException(status_code=400,detail='Unknown model ID')
    if not lock.acquire(blocking=False): raise HTTPException(status_code=429,detail='GPU busy; retry shortly')
    try:
        import __main__
        mdl=getattr(__main__,'model',None)
        tok=getattr(__main__,'tokenizer',None)
        if (mdl is None or tok is None) and not any(m.images() for m in req.messages):
            raise HTTPException(status_code=503,detail='Model not loaded')
        kwargs={'max_new_tokens':req.max_tokens}
        if req.temperature>0: kwargs.update(do_sample=True,temperature=req.temperature)
        else: kwargs['do_sample']=False
        if any(m.images() for m in req.messages):
            response = vision_reply(req.messages, kwargs)
        else:
            messages=[{'role':m.role,'content':m.text()} for m in req.messages]
            encoded=tok.apply_chat_template(messages,add_generation_prompt=True,return_tensors='pt',return_dict=True).to(mdl.device)
            with torch.inference_mode(): out=mdl.generate(**encoded,pad_token_id=tok.eos_token_id,**kwargs)
            response=tok.decode(out[0][encoded['input_ids'].shape[-1]:],skip_special_tokens=True)
        return {'id':'chatcmpl-'+secrets.token_hex(7),'object':'chat.completion',
                'created':int(time.time()),'model':model_id(),
                'choices':[{'index':0,'message':{'role':'assistant','content':response},'finish_reason':'stop'}]}
    finally: lock.release()
