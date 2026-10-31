# PSC Bridges-2 smoke test (1× L40S, Qwen3-0.6B)

Goal: prove the whole pipeline works on one GPU before we touch the TPU or A100.
Two vLLM servers share one GPU (`:8000` plays "tpu", `:8001` plays "a100"); the router
runs every policy on synthetic agent workflows and we print the summary.

Model: **Qwen/Qwen3-0.6B** (smallest Qwen3, same family as the Qwen3-4B we use later,
not gated so no HuggingFace login). Run in float16.

GPU: **L40S** (`l40s-48`) by default, with the latest vLLM. V100 is supported as a fallback
(`GPU_TYPE=v100`), but on V100 the prefix-cache kernel fails to compile, so it is only
good for checking that servers start.

## 1. Setup (once, on a login node)

```bash
ssh <psc_user>@bridges2.psc.edu
cd $PROJECT && git clone https://github.com/clemrho/hetero-agent-routing && cd hetero-agent-routing
WITH_PS=1 bash scripts/psc/setup.sh     # ~10 min; drop WITH_PS=1 to skip the Production Stack router
```

Everything goes into `$PROJECT/har/` (venvs, Python, model weights), not `$HOME`.
The L40S venv is `$PROJECT/har/vllm_env_latest`; the V100 one is `$PROJECT/har/vllm_env`.

## 2. Run

Either interactively:

```bash
interact -p GPU-shared --gres=gpu:l40s-48:1 -t 01:00:00
cd $PROJECT/hetero-agent-routing
bash scripts/psc/run_smoke.sh
```

or as a batch job:

```bash
sbatch scripts/psc/smoke.sbatch        # add -A <allocation> if you have several
squeue -u $USER                        # wait for it
tail -f slurm-har-<jobid>.out
```

Results land in `logs/psc_<jobid>/`: `vllm_*.log`, `check.txt`, `logs/log_<POLICY>_4.jsonl`, `summary.txt`.

## 3. What "passing" looks like

- `[2/5]`: both backends say `OK`, and `warm` has `cached_tokens > 0` and is faster than `cold`.
- `[3/5]` (if installed): the PS router answers on `:8080`.
- `[5/5]`: one block per policy. Expected pattern:
  - `CA`, `PS`: `wf_switched=0/4` (each workflow stays on one backend) and high `cache_hit`
  - `LB`, `RR`: workflows switch backends, so `cache_hit` is lower
  - `FF`: everything on `a100` at this low load

Both backends are the same GPU, so latency differences between policies mean little here.

## Knobs

| Variable | Default | Meaning |
|---|---|---|
| `GPU_TYPE` | `l40s` | `l40s`: latest vLLM, V1 engine. `v100`: vLLM 0.10.1.1, V0 engine, separate venv |
| `MODEL` | `Qwen/Qwen3-0.6B` | e.g. `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B` (run `setup.sh` again to download) |
| `MAX_LEN` | `8192` | `--max-model-len`; Qwen3 supports up to 32768 |
| `GPU_MEM` | `0.4` | GPU memory fraction per backend (two share one GPU) |
| `N_WF` | `4` | concurrent workflows |
| `POLICIES` | `LB FF CA RR` | PS is added automatically if the router is installed |
| `PORT_A` / `PORT_B` / `PS_PORT` | `8000` / `8001` / `8080` | change if someone else on the node uses them |
| `VLLM_VERSION` | from `GPU_TYPE` | vLLM version in `setup.sh`; empty = latest |
| `VENV` | from `GPU_TYPE` | which venv to install into / run from |
| `VLLM_EXTRA_ARGS` | — | extra flags for `vllm serve` |

V100 fallback:

```bash
GPU_TYPE=v100 bash scripts/psc/setup.sh
interact -p GPU-shared --gres=gpu:v100-32:1 -t 01:00:00
GPU_TYPE=v100 bash scripts/psc/run_smoke.sh
# batch: sbatch --gpus=v100-32:1 --export=ALL,GPU_TYPE=v100 scripts/psc/smoke.sbatch
```

Single server by hand: `bash scripts/psc/serve.sh 8000 0.85`, then
`$PROJECT/har/vllm_env_latest/bin/python scripts/check_backend.py --url http://localhost:8000`.

## If it breaks

| Symptom (in `logs/psc_*/vllm_8000.log`) | Try |
|---|---|
| `FileNotFoundError: 'ninja'` / `FlashInfer kernels are disabled` | Re-run `setup.sh`: it installs `ninja` and runs `vllm download-kernels`. Still failing → `module load cuda` so vLLM can compile kernels |
| Second server fails with not enough free memory | `GPU_MEM=0.35` |
| `Port 8000 is already in use` | `PORT_A=8100 PORT_B=8101 PS_PORT=8180 bash scripts/psc/run_smoke.sh` |
| `--enable-prompt-tokens-details` unknown | `VLLM_EXTRA_ARGS` can't remove flags: delete that line in `env.sh`; `cached_tokens` will show as `None` |
| Model download fails on the compute node | run `setup.sh` on the login node first (it pre-downloads the weights) |
| V100: `no kernel image is available`, `torch ... +cu130 does not include kernels for this GPU` | You ran the latest vLLM on V100. Use `GPU_TYPE=v100` for both setup and run |
| V100: `Qwen2Tokenizer has no attribute all_special_tokens_extended` | transformers 5 is too new for vLLM 0.10.1.1. Re-run `GPU_TYPE=v100 bash scripts/psc/setup.sh` (installs `transformers<5`) |
| V100: `prefix_prefill.py ... PassManager::run failed` on the cache-hit request | Known: V100 can't compile the prefix-cache kernel. Use L40S |
