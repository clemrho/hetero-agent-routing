#!/usr/bin/env bash
# One-time setup. Run on a Bridges-2 LOGIN node (it needs internet), from the repo root:
#   bash scripts/psc/setup.sh            # vLLM env + model
#   WITH_PS=1 bash scripts/psc/setup.sh  # also install the Production Stack router
# Pin a vLLM version with VLLM_VERSION=0.x.y if the latest one fails on V100.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
mkdir -p "$HAR_ROOT"
echo "== Installing into $HAR_ROOT"

# 1. uv (fast pip + its own Python 3.12, so we don't depend on cluster modules)
if ! command -v uv >/dev/null; then
  echo "== Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# 2. vLLM venv
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python 3.12 "$VENV"
fi
echo "== Installing vllm${VLLM_VERSION:+==$VLLM_VERSION} (a few minutes)"
uv pip install --python "$VENV/bin/python" "vllm${VLLM_VERSION:+==$VLLM_VERSION}" httpx requests
"$VENV/bin/python" -c "import vllm, torch; print('   vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"

# 3. Model weights (Qwen3 is not gated, no HF login needed)
echo "== Downloading $MODEL to $HF_HOME"
"$VENV/bin/python" -c "from huggingface_hub import snapshot_download as d; print('   ', d('$MODEL'))"

# 4. Optional: official vLLM Production Stack router, in its own venv to avoid dependency clashes
if [[ "${WITH_PS:-0}" == 1 ]]; then
  echo "== Installing Production Stack router into $ROUTER_VENV"
  [[ -d "$HAR_ROOT/production-stack" ]] || \
    git clone --depth 1 https://github.com/vllm-project/production-stack "$HAR_ROOT/production-stack"
  [[ -x "$ROUTER_VENV/bin/python" ]] || uv venv --python 3.12 "$ROUTER_VENV"
  uv pip install --python "$ROUTER_VENV/bin/python" -e "$HAR_ROOT/production-stack"
  echo "   routing-logic options:"
  "$ROUTER_VENV/bin/vllm-router" --help | grep -A3 -- "--routing-logic" || true
fi

echo "== Setup done. Next: get a V100 node and run scripts/psc/run_smoke.sh (see scripts/psc/README.md)"
