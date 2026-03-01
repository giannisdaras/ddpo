#!/bin/bash
# Bootstrap DDPO-JAX on TRC v6e-8 (europe-west4-a, single host, 8 chips)
# Run once after the VM is provisioned.
set -e

echo "=== Installing system deps ==="
sudo apt-get update -q
sudo apt-get install -y -q tmux git python3-pip python3-venv ffmpeg

echo "=== Creating virtualenv ==="
python3 -m venv ~/ddpo-env
source ~/ddpo-env/bin/activate

echo "=== Installing JAX with TPU support ==="
pip install -U pip
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

echo "=== Installing ML deps ==="
# flax 0.7.x: last series that still has jax_utils + training.checkpoints
# (removed in 0.8.0); compatible with JAX 0.9 on v6e
pip install "flax==0.7.5" "optax==0.1.5"
pip install "numpy<2"
pip install "diffusers[flax]==0.12.1" "transformers==4.28.1" "huggingface_hub==0.16.4"
# CPU torch for reward models (CLIP aesthetic scorer etc.)
pip install "torch==2.1.0+cpu" "torchvision==0.16.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu

echo "=== Installing utilities ==="
pip install imageio==2.22.4 tqdm matplotlib h5py
pip install "datasets==2.7.1" gitpython google-cloud-storage gcsfs fsspec
pip install "typed-argument-parser==1.7.2" wandb

echo "=== Cloning fork ==="
if [ ! -d ~/ddpo-jax ]; then
    git clone https://github.com/giannisdaras/ddpo ~/ddpo-jax
fi
cd ~/ddpo-jax
git pull --ff-only
pip install -e .

echo "=== Done. Activate with: source ~/ddpo-env/bin/activate ==="
