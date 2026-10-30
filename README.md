# hetero-agent-routing

Routing agent workflows across heterogeneous accelerators (A100 GPU + TPU) with vLLM.
CMU 11-868 LLM Systems course project.

## Layout

| Path | What |
|---|---|
| `router/router.py` | Router with policies LB / FF / CA / RR / PS, plus a synthetic smoke test |
| `baselines/prodstack/` | Official vLLM Production Stack router (session mode) as the baseline |
| `traces/` | Agent workflow traces; `gen_synthetic.py` makes the synthetic set |
| `replay/` | `replay.py` replays traces through the router; `mock_vllm.py` fake server for testing |
| `analysis/analyze.py` | TTFT, makespan, cache hit rate, device split from the logs |
| `scripts/check_backend.py` | Health + prefix-cache check for any vLLM backend |
| `scripts/psc/` | One-command smoke test on a PSC V100 ([README](scripts/psc/README.md)) |
| `docs/PLAN.md` | Roles, milestones, workflow |

## Quick start (TPU notebook)

```bash
pip install -r requirements.txt
export TPU_URL=http://localhost:8000 A100_URL=http://localhost:8001   # A100 via SSH tunnel
python router/router.py --policy CA --n_wf 4
python analysis/analyze.py logs/log_CA_4.jsonl
```

Production Stack baseline: see [baselines/prodstack/README.md](baselines/prodstack/README.md).
