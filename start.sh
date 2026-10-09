#!/usr/bin/env bash
# Launch Cognivo on a Linux cloud GPU box (RunPod, Lambda, Vast.ai, AWS, GCP...).
#   MODEL=Qwen/Qwen2.5-7B-Instruct bash start.sh
# Then open http://<server-ip>:8080   (set UI_PASSWORD=... to allow remote access)
set -e
cd "$(dirname "$0")"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"   # ~16GB VRAM. 24GB+: Qwen2.5-14B-Instruct-AWQ, 80GB: Llama-3.3-70B-Instruct-AWQ
UI_PORT="${UI_PORT:-8080}"

python3 -m pip install -q --upgrade vllm

# Model server (OpenAI-compatible API) on localhost:8000
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --host 127.0.0.1 --port 8000 \
  --served-model-name "$(basename "$MODEL")" \
  --max-model-len "${MAX_LEN:-8192}" --gpu-memory-utilization 0.90 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT

echo "Waiting for model to load..."
until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM exited"; exit 1; }
  sleep 3
done

python3 server.py --port "$UI_PORT" --backend http://127.0.0.1:8000
