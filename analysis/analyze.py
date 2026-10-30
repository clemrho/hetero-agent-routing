"""
Summarize router logs.

    python analysis/analyze.py logs/log_*.jsonl

For each log file, prints:
    calls, makespan (first arrival -> last done)
    TTFT p50 / p99 (t_first - t_arrive, includes queueing in the router)
    mean workflow completion time
    prefix cache hit rate (sum cached_tokens / sum prompt_tokens)
    device split, and how many workflows switched devices between steps
"""
import json
import sys
from collections import Counter, defaultdict


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def summarize(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    if not recs:
        return None
    ttft = [r["t_first"] - r["t_arrive"] for r in recs if r.get("t_first")]
    by_wf = defaultdict(list)
    for r in recs:
        by_wf[r["wf_id"]].append(r)
    wf_time = [max(x["t_done"] for x in rs) - min(x["t_arrive"] for x in rs) for rs in by_wf.values()]
    switched = sum(len({x["dev"] for x in rs}) > 1 for rs in by_wf.values())
    prompt = sum(r.get("prompt_tokens") or 0 for r in recs)
    cached = sum(r.get("cached_tokens") or 0 for r in recs)
    return {
        "file": path,
        "policy": recs[0]["policy"],
        "calls": len(recs),
        "makespan_s": max(r["t_done"] for r in recs) - min(r["t_arrive"] for r in recs),
        "ttft_p50_s": pct(ttft, 50),
        "ttft_p99_s": pct(ttft, 99),
        "wf_mean_s": sum(wf_time) / len(wf_time),
        "cache_hit": cached / prompt if prompt else float("nan"),
        "devices": dict(Counter(r["dev"] for r in recs)),
        "wf_switched": f"{switched}/{len(by_wf)}",
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        s = summarize(p)
        if s is None:
            print(f"{p}: empty")
            continue
        print(
            f"{s['policy']:>3} {s['file']}\n"
            f"    calls={s['calls']} makespan={s['makespan_s']:.1f}s wf_mean={s['wf_mean_s']:.1f}s\n"
            f"    ttft p50={s['ttft_p50_s']:.3f}s p99={s['ttft_p99_s']:.3f}s cache_hit={s['cache_hit']:.1%}\n"
            f"    devices={s['devices']} wf_switched={s['wf_switched']}"
        )
