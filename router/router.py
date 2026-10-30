"""
Minimal router + baselines + synthetic smoke test.

Policies
    LB  least-loaded (fewest in-flight requests), tie -> FAST
    FF  fast-first: use FAST until it has ff_threshold in flight, then spill
    CA  cache-affinity: step 0 picks by LB, later steps stick to that device
    RR  round-robin
    PS  official vLLM Production Stack router (session mode). We only forward
        to it with header x-session-id = wf_id; it picks the backend itself.
        See baselines/prodstack/.

Usage (inside the TPU notebook):
    pip install -r requirements.txt
    python router/router.py --policy LB --n_wf 4
    # -> logs/log_LB_4.jsonl, one line per LLM call

Backends come from env vars so nobody has to edit this file:
    A100_URL  (default http://localhost:8001, e.g. an SSH tunnel)
    TPU_URL   (default http://localhost:8000)
    PS_URL    (default http://localhost:8080, the Production Stack router)
    MODEL     (default Qwen/Qwen3-4B)

The router is a Python object, not an HTTP proxy: the replay client calls
router.call(...) directly. Timing is always done here, on the client side,
so every policy (including PS) is measured the same way.
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import httpx

BACKENDS = {
    "a100": os.environ.get("A100_URL", "http://localhost:8001"),
    "tpu": os.environ.get("TPU_URL", "http://localhost:8000"),
}
PS_URL = os.environ.get("PS_URL", "http://localhost:8080")
FAST = "a100"  # which backend counts as the fast one (for FF / tie-breaks)
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-4B")
POLICIES = ("LB", "FF", "CA", "RR", "PS")


class Router:
    def __init__(self, policy, log_path, ff_threshold=4):
        assert policy in POLICIES, policy
        self.policy = policy
        self.ff_threshold = ff_threshold          # FF: spill once FAST has this many in flight
        self.inflight = {k: 0 for k in BACKENDS}  # requests currently outstanding per backend
        self.home = {}                            # CA: wf_id -> backend chosen at step 0
        self.rr = 0
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log_path, "a")
        self.client = httpx.AsyncClient(timeout=None)

    def _least_loaded(self):
        return min(self.inflight, key=lambda k: (self.inflight[k], k != FAST))

    def pick(self, wf_id):
        if self.policy == "LB":
            return self._least_loaded()
        if self.policy == "FF":
            if self.inflight[FAST] < self.ff_threshold:
                return FAST
            return min((k for k in BACKENDS if k != FAST), key=lambda k: self.inflight[k])
        if self.policy == "RR":
            names = list(BACKENDS)
            self.rr += 1
            return names[(self.rr - 1) % len(names)]
        if self.policy == "CA":
            if wf_id not in self.home:
                self.home[wf_id] = self._least_loaded()
            return self.home[wf_id]
        return "ps"  # PS: the Production Stack router decides

    async def call(self, wf_id, step, prompt, max_tokens):
        t_arrive = time.time()
        dev = self.pick(wf_id)
        url = PS_URL if dev == "ps" else BACKENDS[dev]
        headers = {"x-session-id": str(wf_id)}
        if dev in self.inflight:
            self.inflight[dev] += 1
        t_send = time.time()
        t_first, usage = None, None
        body = {
            "model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
            "ignore_eos": True, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True},
        }
        try:
            async with self.client.stream(
                "POST", f"{url}/v1/completions", json=body, headers=headers
            ) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    d = json.loads(line[6:])
                    if t_first is None and d.get("choices"):
                        t_first = time.time()
                    if d.get("usage"):
                        usage = d["usage"]
        finally:
            if dev in self.inflight:
                self.inflight[dev] -= 1
        t_done = time.time()
        usage = usage or {}
        details = usage.get("prompt_tokens_details") or {}
        rec = {
            "policy": self.policy, "wf_id": wf_id, "step": step, "dev": dev,
            "t_arrive": t_arrive, "t_send": t_send, "t_first": t_first, "t_done": t_done,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens"),
            "output_tokens": usage.get("completion_tokens"),
        }
        self.log.write(json.dumps(rec) + "\n")
        self.log.flush()
        return rec

    async def close(self):
        await self.client.aclose()
        self.log.close()


# ---------------- synthetic smoke test (until real traces are ready) ----------
async def synthetic_workflow(router, wf_id, steps=5, ctx=2000, add=500, tool_s=1.0, out=32):
    prompt = f"[{wf_id}] " + "hello " * ctx          # unique prefix per workflow
    for s in range(steps):
        await router.call(wf_id, s, prompt, max_tokens=out)
        await asyncio.sleep(tool_s)                    # pretend to run a tool
        prompt += f" [obs {s}] " + "obs " * add        # context grows every step


async def main(policy, n_wf):
    router = Router(policy, f"logs/log_{policy}_{n_wf}.jsonl")
    t0 = time.time()
    await asyncio.gather(*(synthetic_workflow(router, f"wf{i}") for i in range(n_wf)))
    print(f"{policy} n_wf={n_wf}: {time.time() - t0:.1f}s total")
    await router.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="LB", choices=POLICIES)
    ap.add_argument("--n_wf", type=int, default=4)
    a = ap.parse_args()
    asyncio.run(main(a.policy, a.n_wf))
