# Baseline: vLLM Production Stack router (session mode)

The full Production Stack needs Kubernetes + Helm, which the TPU notebook doesn't have.
Its router is a plain Python service, though, and runs on its own with static backends.
That gives us the official implementation as the baseline instead of our own copy of it.

```
TPU notebook
 ├─ vLLM-TPU          localhost:8000
 ├─ vllm-router       localhost:8080   ← start_router.sh, --routing-logic session
 └─ router/router.py --policy PS ──► localhost:8080 (header x-session-id = wf_id)
                          └─► A100 vLLM (public address or SSH tunnel → localhost:8001)
```

## Steps

1. Install the router (see the comment at the top of `start_router.sh`) and run `vllm-router --help`.
2. `bash baselines/prodstack/start_router.sh`
3. `python router/router.py --policy PS --n_wf 4`
4. Check the router's log: every request with the same `wf_id` should land on the same backend.
   Our client log records `dev = "ps"` because the router makes the choice, so this check
   has to come from the router's own log.

## Notes

- Session mode hashes the session id to a backend (probably consistent hashing; not confirmed
  in the docs). That is not the same as our CA policy, which picks the least-loaded device at step 0.
  The report should call this baseline "vLLM Production Stack router, session mode".
- `LOGIC=roundrobin bash start_router.sh` gives a second comparison for free.
- prefix-aware and KV-cache-aware routing need Helm/K8s/LMCache, so we skip them.
- The router only forwards. TTFT and `cached_tokens` are still recorded by `router.py` on the
  client side (streaming passes through unchanged), so the log format is the same for every policy.
