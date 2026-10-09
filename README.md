# Cognivo Cloud — Kaggle AI starter

A working static chat page + connection manager + OpenAI-compatible Kaggle inference endpoint for Qwen2.5-Coder-1.5B-Instruct.

## Quick start (main UI: https://xrenes.github.io/cognivo/)
1. Kaggle notebook settings: **GPU T4 x2**, **Internet On**.
2. Cell 1: paste `kaggle/load_model.py` (set `MODEL_ID`). Cell 2: `%%writefile /kaggle/working/kaggle_server.py` + contents of `kaggle/server.py`. Cell 3: paste `kaggle/start_in_kaggle.py`.
3. Cell 3 prints **1. API Base URL**, **2. Model ID**, **3. API Access Token**.
4. Open the site → **Settings** (sidebar) → paste them into the matching fields → **Test Connection** → **Save Settings**.
5. Switch models: change `MODEL_ID` in cell 1 and rerun only cell 1, then Test Connection again.

`index.html` (root) is the main Cognivo UI. `web/index.html` is an older minimal alternative.

## What is included
- `web/index.html` — browser chat UI, conversation history saved locally, connection settings, test button, link sharing **without** access tokens.
- `kaggle/server.py` — FastAPI bearer-token protected `/v1/models` and `/v1/chat/completions` endpoints, GPU-use lock, CORS allowed origins.
- `kaggle/start_in_kaggle.py` — commands for a Kaggle notebook cell (not a standalone Python script; `%pip` is notebook syntax).

## Exact setup (Kaggle)

1. Open your GPU-enabled Kaggle notebook. The existing Qwen model and tokenizer **must already be loaded** in the current live session as variables `model` and `tokenizer`.
2. Click **Add data → Upload** (or notebook file upload) and upload `kaggle/server.py` renamed to **`kaggle_server.py`**. Its runtime path must be `/kaggle/working/kaggle_server.py`. You can instead create this file with `%%writefile /kaggle/working/kaggle_server.py` and paste in the file's contents.
3. Create a **new notebook code cell** and paste the **contents** of `kaggle/start_in_kaggle.py` into the cell. Check `COGNIVO_ALLOWED_ORIGINS` first: use `https://xrenes.github.io` only if the web page is deployed there. For localhost local dev, set `http://localhost:8080`. It must match the **origin**, not include `/cognivo/`.
4. Run the cell. It prints a random `ACCESS TOKEN` and an `API BASE URL` like `https://random.trycloudflare.com/v1`. If the URL is not ready, run `print(open('/tmp/cognivo_tunnel.log').read())` after the tunnel starts.
5. Keep your Kaggle session running. Do not restart the kernel: otherwise, reload the model and rerun the server cell.

## Run the web app

1. Download/unzip project to PC. Start a local static web server in the directory containing `web/index.html`:
   `cd web && python -m http.server 8080`
2. Open `http://localhost:8080`. If testing locally, set `COGNIVO_ALLOWED_ORIGINS=http://localhost:8080` in Kaggle **before** running the server cell.
3. Select **Connection**, set API base URL, model exactly `Qwen/Qwen2.5-Coder-1.5B-Instruct`, and the private ACCESS TOKEN; select Save and Test.
4. Open Chat and send a message. If you deploy `web/index.html` using GitHub Pages, use that website origin in Kaggle's CORS settings.
5. Use **Copy setup link (no token)** to give a friend the site URL + endpoint/model. Send a token separately over a private channel. The friend's browser must enter it in Connection settings.

## Publish to GitHub Pages

For a separate repo: copy `web/index.html` into a repository root as `index.html`; enable Pages: Settings → Pages → Deploy from branch → main / root. For an existing Cognivo repo, integrate the page/JS or deploy this page at a new route. This package does **not** modify the live Cognivo site.

## Important limitations and security

- This is a **prototype**, not secure multi-user production hosting: testers who know the shared server token have equivalent access. There is no separate login, per-user quota, centralized saved conversations, or permanent connection registry.
- LocalStorage retains chat content in the browser. API token stays in memory unless the user explicitly checks “Remember token”, in which case it is saved unencrypted in localStorage. Use unique, limited-purpose access secrets—not account API credentials.
- Don't post the access token in chat, GitHub, or a screenshot; rotate it by restarting the notebook server with a new random token.
- The Cloudflare Quick Tunnel URL is public and temporary, and can change every session. Token authentication protects the inference endpoints but does not provide full abuse protection. Do not use the public test tunnel for sensitive projects.
- Kaggle may limit or interrupt external networking, tunnel use, GPU access, and session duration. Follow Kaggle rules. The API is available only while the notebook and tunnel run.
- If a browser reports a CORS error, ensure `COGNIVO_ALLOWED_ORIGINS` exactly matches the website origin and restart the backend. The localhost example differs from GitHub Pages.
- For a true product: permanent inference hosting, authentication and user roles, database for chat history, secret management, API gateway with per-user quotas, logging, and abuse controls.
