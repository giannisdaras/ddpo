#!/bin/bash
# Launch DDPO policy-gradient on TRC v6e-8.
# Usage: bash scripts/run_trc_v6e.sh [extra args passed to policy_gradient.py]
set -e

cd ~/ddpo-jax
source ~/ddpo-env/bin/activate

# Load secrets from .env
if [ -f ~/.env ]; then
    export $(grep -v '^#' ~/.env | xargs)
fi

echo "=== JAX devices ==="
python -c "import jax; print(jax.devices())"

echo "=== Launching DDPO ==="
python pipeline/policy_gradient.py \
    --config config.trc_v6e \
    --dataset trc_v6e_compressed \
    "$@" \
    2>&1 | tee ~/ddpo_run_$(date +%Y%m%d_%H%M%S).log
