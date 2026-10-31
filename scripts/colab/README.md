# Colab TPU v5e + PSC L40S smoke test

Open the notebook straight from GitHub:
[tpu_psc_smoke.ipynb](https://colab.research.google.com/github/clemrho/hetero-agent-routing/blob/main/scripts/colab/tpu_psc_smoke.ipynb)

```
Colab (TPU v5e)
 ├─ vLLM-TPU            127.0.0.1:8000   "tpu"   (slow)
 ├─ ssh tunnel          127.0.0.1:8001 ──► bridges2.psc.edu ──► <L40S node>:8000   "a100" label (fast)
 └─ router/router.py
```

Colab can't accept incoming connections, so the router lives in Colab and reaches the L40S
by tunnelling out through the PSC login node.

## Steps

1. **PSC, login node, repo root** (env already set up with `scripts/psc/setup.sh`):
   ```bash
   git pull
   sbatch scripts/psc/serve.sbatch
   cat slurm-har-serve-<jobid>.out      # repeat until you see: READY node=gl0xx port=8000
   ```
2. **Colab**: Runtime → Change runtime type → TPU v5e. Open the notebook, set `PSC_USER` and
   `PSC_NODE` (from the READY line) in cell 4, then run all cells. Cell 4 asks for your PSC password.
3. **Done**: `scancel <jobid>` on PSC, and run the last notebook cell to close the tunnel.

## Notes

- Both sides serve `Qwen/Qwen3-0.6B` in bfloat16 with prefix caching, `max_model_len=8192`.
- TTFT on the L40S includes the Colab ⇄ PSC network round trip. That is real for this setup,
  but keep it in mind when comparing devices.
- If the PSC password login is refused (e.g. MFA), use an SSH key instead: create one in Colab
  (`ssh-keygen -t ed25519 -f ~/.ssh/id_psc -N ""`), add the public key at PSC's SSH key manager,
  and replace `sshpass -e ssh` with `ssh -i ~/.ssh/id_psc` in cell 4.
- The serve job listens on `0.0.0.0` so the login node can forward to it. Other cluster users
  can reach it too, which is fine for a smoke test.
- Port 8000 taken on the node: `sbatch --export=ALL,PORT=8100 scripts/psc/serve.sbatch` and set `PSC_PORT = 8100`.
