# PSC Bridges-2 smoke test (1× V100, Qwen3-0.6B)

Goal: prove the whole pipeline works on one GPU before we touch the TPU or A100.
Two vLLM servers share one V100 (`:8000` plays "tpu", `:8001` plays "a100"); the router
runs every policy on synthetic agent workflows and we print the summary.

Model: **Qwen/Qwen3-0.6B** (smallest Qwen3, same family as the Qwen3-4B we use later,
not gated so no HuggingFace login). Run in float16 because V100 has no bfloat16.

## 1. Setup (once, on a login node)

```bash
ssh <psc_user>@bridges2.psc.edu
cd $PROJECT && git clone https://github.com/clemrho/hetero-agent-routing && cd hetero-agent-routing
WITH_PS=1 bash scripts/psc/setup.sh     # ~10 min; drop WITH_PS=1 to skip the Production Stack router
```

Everything goes into `$PROJECT/har/` (venvs, Python, model weights), not `$HOME`.

## 2. Run

Either interactively:

```bash
interact -p GPU-shared --gres=gpu:v100-32:1 -t 01:00:00
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
| `MODEL` | `Qwen/Qwen3-0.6B` | e.g. `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B` (run `setup.sh` again to download) |
| `MAX_LEN` | `8192` | `--max-model-len`; Qwen3 supports up to 32768 |
| `GPU_MEM` | `0.4` | GPU memory fraction per backend (two share one GPU) |
| `N_WF` | `4` | concurrent workflows |
| `POLICIES` | `LB FF CA RR` | PS is added automatically if the router is installed |
| `VLLM_VERSION` | `0.10.1.1` | vLLM version in `setup.sh` (newer ones don't run on V100, see below) |
| `VLLM_USE_V1` | `0` | `0` = older V0 engine (most reliable on V100 with vLLM 0.10.x) |
| `VLLM_EXTRA_ARGS` | — | extra flags for `vllm serve` |

Single server by hand: `bash scripts/psc/serve.sh 8000 0.85`, then
`$PROJECT/har/vllm_env/bin/python scripts/check_backend.py --url http://localhost:8000`.

## If it breaks

| Symptom (in `logs/psc_*/vllm_8000.log`) | Try |
|---|---|
| `no kernel image is available for execution on the device`, `torch ... +cu130 does not include kernels for this GPU` | The venv has a vLLM built with CUDA 13 (no V100 support). Re-run `bash scripts/psc/setup.sh`: it pins 0.10.1.1 and rebuilds the venv. `setup.sh` prints `archs`, which must include `sm_70` |
| `Qwen2Tokenizer has no attribute all_special_tokens_extended` | transformers 5 is too new for vLLM 0.10.1.1. Re-run `bash scripts/psc/setup.sh` (it now installs `transformers<5` and dependencies as of 2025-09-01) |
| 0.10.1.1 also fails | `VLLM_VERSION=0.9.2 bash scripts/psc/setup.sh` |
| Attention backend error with `VLLM_USE_V1=1` | keep the default `VLLM_USE_V1=0`, or `export VLLM_ATTENTION_BACKEND=TRITON_ATTN` |
| Second server fails with not enough free memory | `GPU_MEM=0.35` |
| `--enable-prompt-tokens-details` unknown | `VLLM_EXTRA_ARGS` can't remove flags: delete that line in `env.sh`; `cached_tokens` will show as `None` |
| Model download fails on the compute node | run `setup.sh` on the login node first (it pre-downloads the weights) |
