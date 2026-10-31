"""
Fake vLLM server for testing the trace -> replay -> router -> log -> analyze pipeline
without a TPU or GPU. NOT for any reported numbers.

    python replay/mock_vllm.py --port 8001 --prefill_tps 20000 --tpot 0.010   # "GPU"
    python replay/mock_vllm.py --port 8000 --prefill_tps 8000  --tpot 0.020   # "TPU"

What it imitates (roughly):
    - streaming /v1/completions with usage.prompt_tokens_details.cached_tokens
    - prefix caching on 16-token blocks (hash chain, like vLLM), LRU-evicted at --cache_tokens
    - one prefill at a time (so queueing shows up), prefill time = uncached tokens / prefill_tps
    - decode at --tpot seconds per output token, ignore_eos honoured
"""
import argparse
import asyncio
import hashlib
import json
import time
from collections import OrderedDict

from aiohttp import web

BLOCK = 16


def block_hashes(ids):
    h, out = b"", []
    for i in range(0, len(ids) - len(ids) % BLOCK, BLOCK):
        h = hashlib.blake2b(h + repr(ids[i:i + BLOCK]).encode(), digest_size=8).digest()
        out.append(h)
    return out


class Engine:
    def __init__(self, a):
        self.a = a
        self.cache = OrderedDict()  # block hash -> None, LRU order
        self.prefill_lock = asyncio.Lock()

    def lookup_and_insert(self, ids):
        hs = block_hashes(ids)
        hit = 0
        for h in hs:
            if h in self.cache:
                hit += 1
                self.cache.move_to_end(h)
            else:
                break
        for h in hs:
            self.cache[h] = None
            self.cache.move_to_end(h)
        while len(self.cache) * BLOCK > self.a.cache_tokens:
            self.cache.popitem(last=False)
        return hit * BLOCK


async def completions(request):
    eng = request.app["engine"]
    body = await request.json()
    prompt = body["prompt"]
    ids = prompt if isinstance(prompt, list) else prompt.split()
    n_out = int(body.get("max_tokens", 16))

    async with eng.prefill_lock:
        cached = eng.lookup_and_insert(ids)
        await asyncio.sleep((len(ids) - cached) / eng.a.prefill_tps + 0.005)
    usage = {"prompt_tokens": len(ids), "completion_tokens": n_out,
             "total_tokens": len(ids) + n_out,
             "prompt_tokens_details": {"cached_tokens": cached}}

    def chunk(text, fin=None):
        return {"id": "mock", "object": "text_completion", "created": int(time.time()),
                "model": body.get("model"), "choices": [{"index": 0, "text": text, "finish_reason": fin}]}

    if not body.get("stream"):
        await asyncio.sleep(eng.a.tpot * (n_out - 1))
        out = chunk("x" * n_out, "length")
        out["usage"] = usage
        return web.json_response(out)

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    for i in range(n_out):
        if i:
            await asyncio.sleep(eng.a.tpot)
        await resp.write(f"data: {json.dumps(chunk('x', 'length' if i == n_out - 1 else None))}\n\n".encode())
    if (body.get("stream_options") or {}).get("include_usage"):
        u = chunk("")
        u["choices"], u["usage"] = [], usage
        await resp.write(f"data: {json.dumps(u)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp


async def models(request):
    return web.json_response({"object": "list", "data": [{"id": request.app["model"], "object": "model"}]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prefill_tps", type=float, default=10000, help="prefill tokens/s")
    ap.add_argument("--tpot", type=float, default=0.015, help="seconds per output token")
    ap.add_argument("--cache_tokens", type=int, default=200_000, help="prefix cache capacity")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    a = ap.parse_args()
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["engine"], app["model"] = Engine(a), a.model
    app.router.add_post("/v1/completions", completions)
    app.router.add_get("/v1/models", models)
    web.run_app(app, port=a.port, print=lambda *_: print(f"mock vLLM on :{a.port}", flush=True))


if __name__ == "__main__":
    main()
