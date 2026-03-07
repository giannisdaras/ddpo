#!/bin/bash
# Launch DDPO policy-gradient on TRC v6e-8.
# Usage: bash scripts/run_trc_v6e.sh [DATASET] [extra args passed to policy_gradient.py]
#   DATASET defaults to trc_v6e_compressed
# Examples:
#   bash scripts/run_trc_v6e.sh
#   bash scripts/run_trc_v6e.sh trc_v6e_brightness
#   bash scripts/run_trc_v6e.sh trc_v6e_saturation
set -e

DATASET="${1:-trc_v6e_compressed}"
shift || true  # shift off dataset arg; remaining args forwarded to python

cd ~/ddpo-jax
source ~/ddpo-env/bin/activate

# Load secrets from .env
if [ -f ~/.env ]; then
    export $(grep -v '^#' ~/.env | xargs)
fi

echo "=== JAX devices ==="
python -c "import jax; print(jax.devices())"

echo "=== Launching DDPO: $DATASET ==="
python pipeline/policy_gradient.py \
    --config config.trc_v6e \
    --dataset "$DATASET" \
    "$@" \
    2>&1 | tee ~/ddpo_run_${DATASET}_$(date +%Y%m%d_%H%M%S).log
