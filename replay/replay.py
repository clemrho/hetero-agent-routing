"""
Replay agent traces through router.Router (owner: Hannah).

    python replay/replay.py --trace traces/synthetic.jsonl --policy CA --concurrency 4 --rep 0
    # -> logs/synthetic/CA_c4_r0.jsonl         one line per LLM call (written by Router)
    #    logs/synthetic/CA_c4_r0.meta.json     run settings + wall time
    #    logs/synthetic/CA_c4_r0.errors.jsonl  failed calls (only if any)

    python replay/replay.py --sweep --trace traces/synthetic.jsonl \
        --policies LB FF CA --concurrency_list 1 4 8 16 --reps 2   # whole matrix

How a workflow is replayed
    for each step:  router.call(prompt, max_tokens=output_tokens)  ->  sleep(tool_s)
    The prompt is a list of token ids (vLLM's /v1/completions accepts that), so token
    counts are exact and both devices see byte-identical input. Step k's prompt is
    the first prompt_tokens ids of one per-workflow random sequence, so every step is an
    exact prefix of the next one, just like a real agent's growing context.

Concurrency (closed loop, the default): N workers; each takes the next workflow from the
trace, runs it to the end, then takes another. --mode open instead starts every workflow
at its arrival_s (needs a trace generated with --rate).

Cache isolation between runs
    Every run uses a fresh random salt for the token ids. Lengths are identical across runs,
    content is not, so run B can't get free cache hits from prompts run A already sent.
    No server restart or cache reset needed between policies.
"""
import argparse
import asyncio
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "router"))
import router as router_mod  # noqa: E402

TOKEN_LO, TOKEN_HI = 1000, 150000  # ordinary Qwen3 vocab ids, well below the special tokens (151643+)


def load_trace(path, limit=None):
    wfs = [json.loads(l) for l in open(path) if l.strip()]
    for wf in wfs:
        st = wf["steps"]
        for a, b in zip(st, st[1:]):
            assert b["prompt_tokens"] >= a["prompt_tokens"], f"{wf['wf_id']}: context shrinks"
    return wfs[:limit] if limit else wfs


def token_ids(salt, wf_id, n):
    seed = int.from_bytes(hashlib.sha256(f"{salt}|{wf_id}".encode()).digest()[:8], "little")
    rng = random.Random(seed)
    return [rng.randrange(TOKEN_LO, TOKEN_HI) for _ in range(n)]


async def call_with_retry(router, wf_id, step, prompt, max_tokens, errors, retries=1):
    for attempt in range(retries + 1):
        try:
            return await router.call(wf_id, step, prompt, max_tokens)
        except Exception as e:  # tunnel hiccup, server restart, ...
            errors.write(json.dumps({
                "wf_id": wf_id, "step": step, "attempt": attempt, "t": time.time(),
                "prompt_tokens": len(prompt), "error": f"{type(e).__name__}: {e}"[:500],
            }) + "\n")
            errors.flush()
            if attempt < retries:
                await asyncio.sleep(1.0)
    return None


async def run_workflow(router, wf, salt, errors, progress):
    st = wf["steps"]
    ids = token_ids(salt, wf["wf_id"], st[-1]["prompt_tokens"])
    for k, s in enumerate(st):
        rec = await call_with_retry(router, wf["wf_id"], k, ids[: s["prompt_tokens"]],
                                    s["output_tokens"], errors)
        if rec is None:
            progress["failed"] += 1
            return  # give up on this workflow; the rest of its steps would be meaningless
        if s.get("tool_s"):
            await asyncio.sleep(s["tool_s"])
    progress["done"] += 1


async def warmup(lengths, max_tokens=8):
    """Hit every backend directly (not logged) so TPU/XLA compiles happen before timing."""
    async with httpx.AsyncClient(timeout=None) as c:
        for name, url in router_mod.BACKENDS.items():
            t0 = time.time()
            for n in lengths:
                for rep in range(2):
                    body = {"model": router_mod.MODEL, "prompt": token_ids(f"warmup{rep}", name, n),
                            "max_tokens": max_tokens, "ignore_eos": True, "temperature": 0}
                    r = await c.post(f"{url}/v1/completions", json=body)
                    r.raise_for_status()
            print(f"  warmup {name:>4} ({url}): {time.time() - t0:.1f}s for lengths {lengths}")


async def run_one(a, wfs, policy, conc, rep):
    out_dir = Path(a.log_dir) / Path(a.trace).stem
    name = f"{policy}_c{conc}_r{rep}"
    log_path = out_dir / f"{name}.jsonl"
    if log_path.exists():
        if not a.overwrite:
            print(f"skip {log_path} (exists; --overwrite to redo)")
            return
        log_path.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    salt = f"{name}-{time.time_ns()}"
    router = router_mod.Router(policy, str(log_path), ff_threshold=a.ff_threshold)
    err_path = out_dir / f"{name}.errors.jsonl"
    errors = open(err_path, "w")
    progress = {"done": 0, "failed": 0}
    t0 = time.time()

    async def ticker():
        while True:
            await asyncio.sleep(30)
            print(f"    {name}: {progress['done']}/{len(wfs)} done, {progress['failed']} failed, "
                  f"{time.time() - t0:.0f}s", flush=True)

    tick = asyncio.create_task(ticker())
    try:
        if a.mode == "closed":
            queue = list(wfs)

            async def worker():
                while queue:
                    await run_workflow(router, queue.pop(0), salt, errors, progress)

            await asyncio.gather(*(worker() for _ in range(conc)))
        else:
            async def delayed(wf):
                await asyncio.sleep(max(0.0, wf["arrival_s"] - (time.time() - t0)))
                await run_workflow(router, wf, salt, errors, progress)

            await asyncio.gather(*(delayed(wf) for wf in wfs))
    finally:
        tick.cancel()
        await router.close()
        errors.close()
    wall = time.time() - t0
    if err_path.stat().st_size == 0:
        err_path.unlink()

    meta = {
        "trace": a.trace, "policy": policy, "concurrency": conc, "rep": rep, "mode": a.mode,
        "ff_threshold": a.ff_threshold, "salt": salt, "n_workflows": len(wfs),
        "done": progress["done"], "failed": progress["failed"], "wall_s": round(wall, 2),
        "t_start": t0, "backends": router_mod.BACKENDS, "model": router_mod.MODEL,
    }
    (out_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    flag = "" if progress["failed"] == 0 else f"  !! {progress['failed']} workflows failed, see {err_path}"
    print(f"{name}: {progress['done']}/{len(wfs)} workflows in {wall:.1f}s -> {log_path}{flag}", flush=True)


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", default="traces/synthetic.jsonl")
    ap.add_argument("--policy", default="CA", help=f"one of {router_mod.POLICIES}")
    ap.add_argument("--concurrency", type=int, default=4, help="concurrent workflows (closed loop)")
    ap.add_argument("--rep", type=int, default=0, help="repetition index, only used in the file name")
    ap.add_argument("--sweep", action="store_true", help="run policies x concurrency_list x reps")
    ap.add_argument("--policies", nargs="+", default=None)
    ap.add_argument("--concurrency_list", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--mode", choices=["closed", "open"], default="closed")
    ap.add_argument("--ff_threshold", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only the first N workflows (quick test)")
    ap.add_argument("--warmup", type=int, nargs="*", default=None,
                    help="warm up every backend first, e.g. --warmup 512 1024 2048 4096 8000")
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    wfs = load_trace(a.trace, a.limit)
    calls = sum(len(w["steps"]) for w in wfs)
    print(f"trace {a.trace}: {len(wfs)} workflows, {calls} LLM calls; backends {router_mod.BACKENDS}")

    if a.warmup is not None:
        await warmup(a.warmup or [512, 1024, 2048, 4096])

    if a.sweep:
        policies = a.policies or ["LB", "FF", "CA"]
        for p in policies:
            assert p in router_mod.POLICIES, f"unknown policy {p}; router has {router_mod.POLICIES}"
        # reps outermost, so an interrupted sweep still has one full pass of every config
        for rep in range(a.reps):
            for conc in a.concurrency_list:
                for p in policies:
                    await run_one(a, wfs, p, conc, rep)
    else:
        assert a.policy in router_mod.POLICIES, f"unknown policy {a.policy}"
        await run_one(a, wfs, a.policy, a.concurrency, a.rep)


if __name__ == "__main__":
    asyncio.run(main())
