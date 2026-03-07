#!/bin/bash
# Bootstrap DDPO-JAX on TRC v6e-8 (europe-west4-a, single host, 8 chips)
# Run once after the VM is provisioned.
set -e

echo "=== Installing system deps ==="
sudo apt-get update -q
sudo apt-get install -y -q tmux git python3-pip python3-venv ffmpeg cargo

echo "=== Creating virtualenv ==="
python3 -m venv ~/ddpo-env
source ~/ddpo-env/bin/activate

echo "=== Installing JAX with TPU support ==="
pip install -U pip
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

echo "=== Installing ML deps ==="
# latest flax: compatible with JAX 0.9 on v6e; deprecated APIs shimmed in source
pip install flax "optax==0.1.5"
# diffusers 0.12.1 + transformers 4.36 (pre-built tokenizers wheels for Py3.12)
pip install "diffusers[flax]==0.12.1" "transformers==4.36.2"
# CPU torch for reward models (CLIP aesthetic scorer etc.)
# Must be >=2.4.0: JAX 0.9.1 requires numpy>=2.0, and torch<2.4 was compiled against numpy 1.x
pip install "torch==2.4.0+cpu" "torchvision==0.19.0+cpu" \
    --index-url https://download.pytorch.org/whl/cpu

echo "=== Installing utilities ==="
pip install imageio==2.22.4 tqdm matplotlib h5py
pip install "datasets==2.7.1" gitpython google-cloud-storage gcsfs fsspec
pip install "typed-argument-parser==1.7.2" wandb inflect dill pytz setuptools

echo "=== Cloning fork ==="
if [ ! -d ~/ddpo-jax ]; then
    git clone https://github.com/giannisdaras/ddpo ~/ddpo-jax
fi
cd ~/ddpo-jax
git pull --ff-only
pip install -e .

echo "=== Done. Activate with: source ~/ddpo-env/bin/activate ==="
