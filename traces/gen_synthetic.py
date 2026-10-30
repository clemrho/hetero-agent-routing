"""
Generate synthetic agent-workflow traces (owner: Hannah).

Each workflow is: LLM call -> tool wait -> LLM call -> tool wait -> ...
and every step's prompt = previous prompt + previous output + new tool observation,
so earlier steps are always an exact prefix of later ones.

Defaults follow the project plan:
    steps        {3, 5, 7, 9}
    initial ctx  {1k, 4k} tokens
    tool obs     ~500 tokens appended per step   (uniform 400-600)
    tool wait    0.5 - 2.0 s                     (uniform)
    output len   50 - 200 tokens per step        (uniform; agent actions are short)

Usage:
    python traces/gen_synthetic.py                       # -> traces/synthetic.jsonl
    python traces/gen_synthetic.py --per_combo 8 --max_model_len 16384 --out traces/syn16k.jsonl

Workflows whose context would exceed --max_model_len are cut short at the last step
that fits ("truncated": true). The summary printed at the end shows how many were cut,
so keep --max_model_len equal to vLLM's --max-model-len on BOTH devices.

Output format (one JSON object per line, one line per workflow) is documented in
traces/README.md; replay/replay.py reads it.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def make_workflow(rng, wf_id, n_steps, init_ctx, a):
    steps, ctx, truncated = [], init_ctx, False
    for s in range(n_steps):
        out = rng.randint(a.out_min, a.out_max)
        if ctx + out > a.max_model_len:
            truncated = True
            break
        last = s == n_steps - 1
        obs = 0 if last else rng.randint(a.obs_min, a.obs_max)
        tool_s = 0.0 if last else round(rng.uniform(a.tool_min, a.tool_max), 3)
        steps.append({
            "prompt_tokens": ctx,   # full context sent at this step
            "output_tokens": out,   # max_tokens for this call (replay uses ignore_eos)
            "tool_s": tool_s,       # tool wait after this call (0 on the last step)
            "obs_tokens": obs,      # tool output appended before the next step
        })
        ctx += out + obs
    if steps:
        steps[-1]["tool_s"], steps[-1]["obs_tokens"] = 0.0, 0
    return {
        "wf_id": wf_id,
        "source": "synthetic",
        "arrival_s": 0.0,
        "steps_planned": n_steps,
        "init_ctx": init_ctx,
        "truncated": truncated,
        "steps": steps,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="traces/synthetic.jsonl")
    ap.add_argument("--steps", type=int, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--init_ctx", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--per_combo", type=int, default=4, help="workflows per (steps, init_ctx) combo")
    ap.add_argument("--obs_min", type=int, default=400)
    ap.add_argument("--obs_max", type=int, default=600)
    ap.add_argument("--out_min", type=int, default=50)
    ap.add_argument("--out_max", type=int, default=200)
    ap.add_argument("--tool_min", type=float, default=0.5)
    ap.add_argument("--tool_max", type=float, default=2.0)
    ap.add_argument("--max_model_len", type=int, default=8192, help="must match vLLM --max-model-len")
    ap.add_argument("--rate", type=float, default=0.0,
                    help="workflows/s for open-loop arrivals (Poisson). 0 = all arrive at t=0 "
                         "(closed-loop replay, the default experiment, ignores arrival_s anyway)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    wfs = []
    for n in a.steps:
        for c in a.init_ctx:
            for i in range(a.per_combo):
                wfs.append(make_workflow(rng, f"syn-s{n}-c{c}-{i}", n, c, a))
    rng.shuffle(wfs)  # mix long and short workflows so closed-loop replay isn't ordered by size

    t = 0.0
    for wf in wfs:
        if a.rate > 0:
            t += rng.expovariate(a.rate)
        wf["arrival_s"] = round(t, 3)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for wf in wfs:
            f.write(json.dumps(wf) + "\n")

    # summary
    by = defaultdict(list)
    for wf in wfs:
        by[(wf["steps_planned"], wf["init_ctx"])].append(wf)
    print(f"wrote {len(wfs)} workflows -> {a.out}   (max_model_len={a.max_model_len})")
    print(f"{'steps':>5} {'ctx0':>5} {'n':>3} {'kept steps':>10} {'max ctx':>8} {'truncated':>9}")
    for (n, c), g in sorted(by.items()):
        kept = sorted({len(w['steps']) for w in g})
        mx = max(w["steps"][-1]["prompt_tokens"] + w["steps"][-1]["output_tokens"] for w in g)
        cut = sum(w["truncated"] for w in g)
        print(f"{n:>5} {c:>5} {len(g):>3} {str(kept):>10} {mx:>8} {cut:>6}/{len(g)}")
    calls = sum(len(w["steps"]) for w in wfs)
    tool = sum(s["tool_s"] for w in wfs for s in w["steps"])
    print(f"total LLM calls={calls}, total tool wait={tool:.0f}s "
          f"(lower bound on a concurrency-1 run, before any LLM time)")


if __name__ == "__main__":
    main()
