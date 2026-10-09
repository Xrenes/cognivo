# Cognivo

A self-hosted AI chat UI that talks to any OpenAI-compatible model server (vLLM, Ollama, LM Studio, TGI).

## Run on a cloud GPU
```bash
MODEL=Qwen/Qwen2.5-7B-Instruct bash start.sh
```
Open `http://<server-ip>:8080`. `server.py` serves the UI and proxies `/v1` to vLLM.

## Hosted UI
The UI on GitHub Pages can connect to your GPU server: **Settings → API base URL**.
The URL must be **HTTPS** (an HTTPS page can't call plain `http://`), e.g. a RunPod proxy URL
or a Cloudflare Tunnel, and the server must allow CORS from this origin.
