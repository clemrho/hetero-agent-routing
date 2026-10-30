# Plan and roles

Goal: route agent workflows (multi-step LLM calls with tool gaps in between) across
heterogeneous devices (A100 GPU + TPU), and show that our strategy (working name **GapShift**:
use the tool gap to move a workflow's context to the faster device) beats the baselines on
workflow completion time / TTFT without losing too much prefix-cache reuse.

## Who owns what

Each person owns a directory, so PRs rarely conflict. Names are a proposal — swap as needed.

| Person | Owns | Directory |
|---|---|---|
| Yanze | Router, baselines (LB / FF / CA / RR / PS), GapShift | `router/`, `baselines/` |
| Hannah | Agent traces and the replay client that feeds them to the router | `traces/`, `replay/` |
| Clement | Infra (vLLM-TPU, A100 vLLM, SSH tunnel, Production Stack router), experiment runs, analysis + plots | `scripts/`, `analysis/` |

Shared interfaces (change only with a heads-up to everyone):
- `Router.call(wf_id, step, prompt, max_tokens) -> rec` in `router/router.py`
- The log line format (one JSON per LLM call) — see `analysis/analyze.py`
- The trace format in `traces/README.md`

## Milestones

### Week 1 — everything runs end to end
- [ ] Clement: vLLM-TPU up on :8000, A100 vLLM reachable from the notebook (tunnel → :8001), same model on both
- [ ] Clement: Production Stack router installed, `vllm-router --help` shows `session` and `--session-key`
- [ ] Yanze: `router.py --policy {LB,FF,CA,RR}` smoke test with synthetic workflows on both devices
- [ ] Yanze: `--policy PS` goes through the Production Stack router; confirm from its log that each `wf_id` sticks to one backend
- [ ] Hannah: trace format agreed; first small set of real agent traces (steps, prompt growth, tool durations, output lengths)
- [ ] Clement: `analysis/analyze.py` runs on the smoke-test logs

### Week 2 — baselines on real traces
- [ ] Hannah: `replay/` replays traces through `Router` with real tool gaps and arrival times
- [ ] Clement: sweep n_wf (concurrency) × policy, save logs under `logs/`
- [ ] Yanze: measure per-device prefill/decode speed and cache-transfer (re-prefill) cost → numbers GapShift will use
- [ ] All: first results table + plot, agree on main metric

### Week 3 — GapShift
- [ ] Yanze: GapShift policy — at a tool gap, decide whether to move the workflow to the faster device (cost of re-prefill vs. expected queueing saved)
- [ ] Hannah: tool-duration predictor / trace stats that GapShift needs
- [ ] Clement: rerun the sweep with GapShift, ablations (threshold, gap length)

### Week 4 — write-up
- [ ] Final experiments, plots, report, presentation

## Workflow

- `main` is protected by habit: every change goes through a PR from a personal branch (`yanze/...`, `hannah/...`, `clement/...`).
- One reviewer approves before merge.
- Don't commit model weights or large logs; `logs/` is git-ignored (put final result logs in `results/`).
