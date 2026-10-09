# Paste into a Kaggle notebook code cell AFTER the model-loading cell (load_model.py) has run.
# Create /kaggle/working/kaggle_server.py first (%%writefile + contents of server.py).
%pip install -q fastapi 'uvicorn[standard]'
import os, secrets, subprocess, time, re, requests, threading, uvicorn
# Websites allowed to call this server (origins only, no path).
os.environ['COGNIVO_ALLOWED_ORIGINS'] = 'https://xrenes.github.io,http://localhost:8080'
os.environ['COGNIVO_MODEL_ID'] = MODEL_ID  # set MODEL_ID in your model-loading cell
_first_run = not globals().get('_cognivo_started')  # rerunning this cell just reprints the details
if _first_run:
    os.environ['COGNIVO_API_KEY'] = secrets.token_urlsafe(32)

if _first_run:
    # Notebook model resides in __main__; uvicorn must run IN THIS PROCESS to share it.
    from kaggle_server import app
    threading.Thread(target=lambda: uvicorn.run(app, host='127.0.0.1', port=8000, log_level='warning'), daemon=True).start()
    time.sleep(3)

    # Temporary public HTTPS tunnel (cloudflared, from Cloudflare's official GitHub releases).
    subprocess.run(['wget', '-q', '-O', '/tmp/cloudflared',
                    'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64'], check=True)
    subprocess.run(['chmod', '+x', '/tmp/cloudflared'], check=True)
    log = open('/tmp/cognivo_tunnel.log', 'w')
    process = subprocess.Popen(['/tmp/cloudflared', 'tunnel', '--url', 'http://127.0.0.1:8000', '--no-autoupdate'],
                               stdout=log, stderr=subprocess.STDOUT)

    _cognivo_started = True

base_url = None
for _ in range(60 if _first_run else 1):  # wait up to ~60 s for the tunnel URL
    time.sleep(1)
    found = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', open('/tmp/cognivo_tunnel.log').read())
    if found: base_url = found.group(0) + '/v1'; break

# Check the whole path (tunnel -> server -> model list) before printing.
status = 'tunnel not ready - inspect /tmp/cognivo_tunnel.log'
if base_url:
    for _ in range(20):
        try:
            r = requests.get(base_url + '/models', headers={'Authorization': 'Bearer ' + os.environ['COGNIVO_API_KEY']}, timeout=10)
            status = 'OK - ready to connect' if r.ok else f'server answered HTTP {r.status_code}'
            if r.ok: break
        except Exception as e:
            status = f'tunnel not reachable yet ({type(e).__name__})'
        time.sleep(3)

print('\n' + '=' * 64)
print(' COGNIVO - paste these into Settings (Kaggle Connection)')
print('=' * 64)
print(' 1. API Base URL   :', base_url or 'NOT READY')
print(' 2. Model ID       :', os.environ['COGNIVO_MODEL_ID'])
print(' 3. API Access Token:', os.environ['COGNIVO_API_KEY'])
print('-' * 64)
print(' Status:', status)
print(' Keep the token private. A new URL + token only after a kernel restart.')
print('=' * 64)
