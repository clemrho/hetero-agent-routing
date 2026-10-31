"""
Check that one vLLM backend works and that prefix caching is on.
Works for any backend (PSC GPU, TPU), standard library only.

    python scripts/check_backend.py --url http://localhost:8000 --model Qwen/Qwen3-0.6B

Prints the served models, then sends the same long prefix twice:
    cold: first time, the whole prompt is prefilled
    warm: same prefix, new suffix -> should be faster with cached_tokens > 0
Exit code 1 if the server is unreachable, the model is missing, or the warm
request got no cached tokens.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def post(url, body, headers=None):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:2000]
        sys.exit(f"FAIL {url}: HTTP {e.code}\n{detail}\n(see the server log for the traceback)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--ctx", type=int, default=3000, help="prefix length in words (~tokens)")
    ap.add_argument("--session", default=None, help="send x-session-id (for the PS router)")
    a = ap.parse_args()
    headers = {"x-session-id": a.session} if a.session else None

    try:
        models = [m["id"] for m in get(f"{a.url}/v1/models")["data"]]
    except Exception as e:  # noqa: BLE001
        sys.exit(f"FAIL {a.url}: cannot reach /v1/models ({e})")
    print(f"{a.url} serves {models}")
    if a.model not in models:
        sys.exit(f"FAIL model {a.model} not served")

    def ttft(prompt):
        t = time.time()
        r = post(f"{a.url}/v1/completions",
                 {"model": a.model, "prompt": prompt, "max_tokens": 1, "temperature": 0},
                 headers)
        u = r.get("usage") or {}
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        return time.time() - t, u.get("prompt_tokens"), cached

    post(f"{a.url}/v1/completions", {"model": a.model, "prompt": "warmup", "max_tokens": 1}, headers)
    prefix = f"[{uuid.uuid4().hex}] " + "word " * a.ctx  # random tag -> cold on every run
    cold = ttft(prefix + "Q1")
    print(f"  cold ok ({cold[0] * 1000:.1f} ms), sending warm request (prefix-cache hit)...")
    warm = ttft(prefix + "Q2")
    for name, (t, n, c) in (("cold", cold), ("warm", warm)):
        print(f"  {name}: {t * 1000:7.1f} ms  prompt_tokens={n}  cached_tokens={c}")

    if warm[2] is None:
        print("WARN cached_tokens not reported (is --enable-prompt-tokens-details set?)")
    elif warm[2] == 0:
        sys.exit("FAIL warm request had 0 cached tokens: prefix caching is not working")
    print(f"OK  speedup cold/warm = {cold[0] / warm[0]:.1f}x")


if __name__ == "__main__":
    main()
