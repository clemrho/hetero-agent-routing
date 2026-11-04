# hetero-agent-routing

Routing agent workflows across heterogeneous accelerators (TPU v5e + GPU) with vLLM.
CMU 11-868 LLM Systems course project · Yanze, Keyi, Hannah.

An agent workflow is a chain of LLM calls with tool calls in between; each call's prompt is the
previous one plus new tokens, so prefix caching saves most of the prefill, but only if the call
lands on the device that holds the cache. With two devices that have different speed profiles,
the router has to trade cache reuse against load. We compare standard policies on this setting
and are building **GapShift**, which uses the tool-call gaps to re-place a workflow's context.

- Midterm report: [docs/midterm_report.md](docs/midterm_report.md)
- Plan, roles, timeline: [docs/PLAN.md](docs/PLAN.md)
- Results: [results/](results/README.md) (latest: [2026-11-03 TPU v5e + L40S, long context to ~27k](results/2026-11-03_tpu-v5e_l40s_long16/README.md))

## How it fits together

```
Colab (TPU v5e)
 ├─ vLLM-TPU               127.0.0.1:8000  "tpu"
 ├─ SSH tunnel             127.0.0.1:8001 ──► bridges2.psc.edu ──► PSC L40S node: vLLM  "gpu"
 └─ replay/replay.py ──► router/router.py (policy) ──► one of the two backends
                         └─ logs one JSON line per call ──► analysis/analyze.py
```

The router runs next to the TPU because the Colab runtime can only open outgoing connections.

## Layout

| Path | What |
|---|---|
| `router/router.py` | Router: policies LB / FF / CA / RR / PS, client-side timing (TTFT, cached tokens), synthetic smoke test |
| `baselines/prodstack/` | Official vLLM Production Stack router in session mode (PS baseline) |
| `traces/` | Trace format; `gen_synthetic.py` generates synthetic agent workflows |
| `replay/` | `replay.py` replays traces through the router (sweeps over policy × concurrency × reps); `mock_vllm.py` fake server for testing |
| `analysis/analyze.py` | Makespan, TTFT p50/p99, cache hit rate vs. its upper bound, device split |
| `scripts/check_backend.py` | Health + cold/warm prefix-cache check for any vLLM backend |
| `scripts/psc/` | PSC Bridges-2: setup, smoke test, `serve.sbatch` for the remote GPU ([README](scripts/psc/README.md)) |
| `scripts/colab/` | Colab TPU v5e notebook: vLLM-TPU, tunnel, checks, baseline sweep ([README](scripts/colab/README.md)) |
| `results/` | One folder per reported run: setup, commands, tables, raw output |
| `docs/` | Plan and reports |

## Policies

| Policy | Rule |
|---|---|
| LB | least in-flight requests (ties → `gpu`) |
| FF | `gpu` until it has 4 in flight, then spill to `tpu` |
| CA | first step by LB, then the workflow stays on that device |
| RR | alternate devices |
| PS | official Production Stack router, session mode (hash of `x-session-id` = workflow id) |
| GapShift | in progress: at each tool gap, re-place the context where the next call is cheapest |

## Run it

1. **GPU on PSC** (once: `bash scripts/psc/setup.sh` on a login node, add `MODEL=Qwen/Qwen3-4B` for the 4B weights):
   ```bash
   sbatch --export=ALL,MODEL=Qwen/Qwen3-4B,MAX_LEN=16384 scripts/psc/serve.sbatch
   cat slurm-har-serve-<jobid>.out    # wait for: READY node=...
   ```
2. **TPU on Colab**: open [scripts/colab/tpu_psc_smoke.ipynb](https://colab.research.google.com/github/clemrho/hetero-agent-routing/blob/yanze/colab-tpu-psc/scripts/colab/tpu_psc_smoke.ipynb)
   (branch `yanze/colab-tpu-psc` until it is merged) with a TPU v5e runtime; set `BRANCH`, `MODEL`, `MAX_LEN`, `PSC_USER`, `PSC_NODE`, run all cells.
3. **Single GPU only** (no TPU): `bash scripts/psc/run_smoke.sh` on a PSC GPU node runs both backends on one GPU.

Environment variables used by the router: `TPU_URL`, `GPU_URL` (old name `A100_URL` still works), `PS_URL`, `MODEL`.
