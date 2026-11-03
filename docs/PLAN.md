# Plan and roles

Goal: route agent workflows (multi-step LLM calls with tool waits in between) across two
accelerators with different performance profiles (TPU v5e + a GPU, currently a PSC L40S),
each running vLLM with prefix caching. Show that a policy that uses the tool gaps to place a
workflow's context where it is cheapest (working name **GapShift**) beats affinity-only and
load-only baselines on workflow completion time and TTFT.

We do not assume one device is "fast" and the other "slow": measured profiles decide
(see [results/2026-11-03](../results/2026-11-03_tpu-v5e_l40s_quick12/README.md)).

## Who owns what

| | Yanze | Keyi | Hannah |
|---|---|---|---|
| In one line | System + experiments | Device profiling + plots | Workloads + replay |
| Work | vLLM on TPU and GPU, TPU ⇄ GPU connection, router + baselines (LB / FF / CA / RR / PS), sweeps, GapShift | Unloaded TTFT on both devices (cold / warm, 1k–16k tokens), decode speed; Fig. 1 (device profiles), Fig. 2 (workflow completion time), Fig. 3 (latency breakdown) | Synthetic traces, HotpotQA traces, `replay/replay.py` |
| Code | `router/`, `baselines/`, `scripts/` | `analysis/` | `traces/`, `replay/` |
| Hands off | Logs → Keyi | Profile table → GapShift cost model | Traces + replay → Yanze |

Shared interfaces (change only with a heads-up to everyone):
- `Router.call(wf_id, step, prompt, max_tokens) -> rec` in `router/router.py`
- The log line format (one JSON per LLM call), see `analysis/analyze.py`
- The trace format in `traces/README.md`

## Status (2026-11-03)

- [x] vLLM on PSC GPUs (V100 and L40S; V100 can't run prefix caching, see `scripts/psc/README.md`)
- [x] vLLM-TPU on Colab TPU v5e, GPU reached over an SSH tunnel through the PSC login node
- [x] Router with LB / FF / CA / RR, and PS (official Production Stack router, session mode) on PSC
- [x] Synthetic trace generator + replay client (Hannah)
- [x] First cross-device baseline: LB / FF / CA / RR × concurrency {4, 8}, 1 rep ([results](../results/2026-11-03_tpu-v5e_l40s_quick12/README.md))
- [x] Midterm report draft: [docs/midterm_report.md](midterm_report.md)

## Timeline

| Dates | Yanze | Keyi | Hannah |
|---|---|---|---|
| 11/4 | Midterm report due | | |
| 11/5–11/9 | Move the TPU next to PSC (GCP v5e VM in a US-East zone, or the course hub) to remove the ~200 ms tunnel RTT; PS baseline in the cross-device setup | Device profiles on both devices (Fig. 1) | HotpotQA traces (due 11/10) |
| 11/10–11/16 | Medium / long-context sweep: contexts to 16k, concurrency {4, 8, 16}, 2–3 reps; RTT-corrected analysis | Fig. 2, Fig. 3 from the sweep | Replay on HotpotQA traces |
| 11/17–11/26 | Implement GapShift (all three decide the cost model together) | Ablations: gap length, threshold | Tool-duration stats for GapShift |
| 11/27–12/4 | Final runs, report, presentation (12/4) | Final plots | Final traces |

## Workflow

- Every change goes through a personal branch (`yanze/...`, `hannah/...`, `keyi/...`) and a PR; one reviewer approves before merge.
- Don't commit model weights or large logs; `logs/` is git-ignored. Runs we report go in `results/<date>_<devices>_<what>/`.
