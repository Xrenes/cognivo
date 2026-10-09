"""Serves index.html and proxies /v1/* to the model server (vLLM) with streaming.
Stdlib only.  python server.py --port 8080 --backend http://127.0.0.1:8000
"""
import argparse, base64, hmac, http.server, os, urllib.request, urllib.error

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, default=int(os.getenv("UI_PORT", 8080)))
p.add_argument("--backend", default=os.getenv("BACKEND", "http://127.0.0.1:8000"))
args = p.parse_args()
PASSWORD = os.getenv("UI_PASSWORD", "")  # set to require HTTP Basic auth (any username)
ROOT = os.path.dirname(os.path.abspath(__file__))


class H(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def _proxy(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)) or None
        hdrs = {k: v for k, v in self.headers.items() if k.lower() in ("content-type", "authorization")}
        req = urllib.request.Request(args.backend + self.path, data=body, headers=hdrs, method=self.command)
        try:
            r = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            r = e
        except Exception as e:
            msg = f"Model backend unavailable: {e}".encode()
            self.send_response(502); self.send_header("Content-Length", str(len(msg))); self.end_headers()
            self.wfile.write(msg); return
        self.send_response(r.status)
        self.send_header("Content-Type", r.headers.get("Content-Type", "application/json"))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        while chunk := r.read1(4096) if hasattr(r, "read1") else r.read(4096):
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk)); self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")

    def _authed(self):
        if not PASSWORD: return True
        h = self.headers.get("Authorization", "")
        try: pw = base64.b64decode(h[6:]).decode().split(":", 1)[1] if h.startswith("Basic ") else ""
        except Exception: pw = ""
        if hmac.compare_digest(pw, PASSWORD): return True
        self.send_response(401); self.send_header("WWW-Authenticate", 'Basic realm="Cognivo"')
        self.send_header("Content-Length", "0"); self.end_headers(); return False

    def do_GET(self):
        if not self._authed(): return
        if self.path.startswith("/v1/"): return self._proxy()
        if self.path in ("/", ""): self.path = "/index.html"
        if not self.path.split("?")[0] in ("/index.html",):
            self.send_error(404); return
        super().do_GET()

    def do_POST(self):
        if not self._authed(): return
        if self.path.startswith("/v1/"): return self._proxy()
        self.send_error(404)


if not PASSWORD: print("WARNING: UI_PASSWORD not set - anyone with the URL can use your GPU")
print(f"Cognivo UI on http://0.0.0.0:{args.port}  ->  model at {args.backend}")
http.server.ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()
