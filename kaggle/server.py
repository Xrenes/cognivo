"""Use in Kaggle after loading global `model` and `tokenizer` objects."""
import os, time, secrets, threading
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
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

class Message(BaseModel):
    role: Literal['system','user','assistant']
    content: str = Field(min_length=1,max_length=12000)
class Completion(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1,max_length=16)
    max_tokens: int = Field(default=512,ge=1,le=1024)
    temperature: float = Field(default=0.3,ge=0,le=2)

def authorize(header):
    prefix='Bearer '
    if not header or not header.startswith(prefix) or not secrets.compare_digest(header[len(prefix):],API_KEY):
        raise HTTPException(status_code=401,detail='Invalid access token')

@app.get('/v1/models')
def models(authorization: str|None=Header(default=None)):
    authorize(authorization)
    return {'object':'list','data':[{'id':model_id(),'object':'model','owned_by':'self'}]}

@app.post('/v1/chat/completions')
def complete(req:Completion,authorization:str|None=Header(default=None)):
    authorize(authorization)
    if req.model!=model_id(): raise HTTPException(status_code=400,detail='Unknown model ID')
    if not lock.acquire(blocking=False): raise HTTPException(status_code=429,detail='GPU busy; retry shortly')
    try:
        import __main__
        mdl=getattr(__main__,'model',None)
        tok=getattr(__main__,'tokenizer',None)
        if mdl is None or tok is None: raise HTTPException(status_code=503,detail='Model not loaded')
        messages=[m.model_dump() for m in req.messages]
        encoded=tok.apply_chat_template(messages,add_generation_prompt=True,return_tensors='pt',return_dict=True).to(mdl.device)
        kwargs={'max_new_tokens':req.max_tokens,'pad_token_id':tok.eos_token_id}
        if req.temperature>0: kwargs.update(do_sample=True,temperature=req.temperature)
        else: kwargs['do_sample']=False
        with torch.inference_mode(): out=mdl.generate(**encoded,**kwargs)
        response=tok.decode(out[0][encoded['input_ids'].shape[-1]:],skip_special_tokens=True)
        return {'id':'chatcmpl-'+secrets.token_hex(7),'object':'chat.completion',
                'created':int(time.time()),'model':model_id(),
                'choices':[{'index':0,'message':{'role':'assistant','content':response},'finish_reason':'stop'}]}
    finally: lock.release()
