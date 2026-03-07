from . import user
from .base import base

# DDPO policy-gradient config tuned for TRC v6e-8 (8 chips, single host)
# sample_batch_size=4 per device x 8 devices = 32 images/batch x 8 batches = 256 samples/epoch
# train_batch_size=1 per device x 8 devices x train_accumulation_steps=8 = 64 effective train batch
# Matches PyTorch reference scale: 256 samples/epoch, 64 effective train batch

_pg_trc = {
    "sample_batch_size": 4,
    "num_sample_batches_per_epoch": 8,
    "train_batch_size": 1,
    "train_accumulation_steps": 8,
    "num_train_epochs": 100,
    "save_freq": 10,
    # LoRA: match PyTorch reference (rank=4, alpha=4, gaussian init)
    "use_lora": True,
    "lora_rank": 4,
    "lora_alpha": 4,
    # Match PyTorch DGX per-prompt buffer size
    "per_prompt_stats_bufsize": 16,
}

trc_v6e_compressed = {
    "common": {
        "logbase": f"{user.bucket}/logs/trc-v6e-compressed",
        "prompt_fn": "imagenet_animals",
        "filter_field": "jpeg",
    },
    "pg": _pg_trc,
}

# LoRA version of compressed — single test run to verify TPU reproduces PyTorch
trc_v6e_compressed_lora = {
    "common": {
        "logbase": f"{user.bucket}/logs/trc-v6e-compressed-lora",
        "prompt_fn": "imagenet_animals",
        "filter_field": "jpeg",
    },
    "pg": _pg_trc,
}

trc_v6e_brightness = {
    "common": {
        "logbase": f"{user.bucket}/logs/trc-v6e-brightness",
        "prompt_fn": "imagenet_animals",
        "filter_field": "brightness",
    },
    "pg": _pg_trc,
}

trc_v6e_saturation = {
    "common": {
        "logbase": f"{user.bucket}/logs/trc-v6e-saturation",
        "prompt_fn": "imagenet_animals",
        "filter_field": "saturation",
    },
    "pg": _pg_trc,
}

trc_v6e_entropy = {
    "common": {
        "logbase": f"{user.bucket}/logs/trc-v6e-entropy",
        "prompt_fn": "imagenet_animals",
        "filter_field": "entropy",
    },
    "pg": _pg_trc,
}

trc_v6e_hue_diversity = {
    "common": {
        "logbase": f"{user.bucket}/logs/trc-v6e-hue-diversity",
        "prompt_fn": "imagenet_animals",
        "filter_field": "hue_diversity",
    },
    "pg": _pg_trc,
}

trc_v6e_color_temperature = {
    "common": {
        "logbase": f"{user.bucket}/logs/trc-v6e-color-temperature",
        "prompt_fn": "imagenet_animals",
        "filter_field": "color_temperature",
    },
    "pg": _pg_trc,
}

trc_v6e_fourier_smoothness = {
    "common": {
        "logbase": f"{user.bucket}/logs/trc-v6e-fourier-smoothness",
        "prompt_fn": "imagenet_animals",
        "filter_field": "fourier_smoothness",
    },
    "pg": _pg_trc,
}

trc_v6e_local_contrast = {
    "common": {
        "logbase": f"{user.bucket}/logs/trc-v6e-local-contrast",
        "prompt_fn": "imagenet_animals",
        "filter_field": "local_contrast",
    },
    "pg": _pg_trc,
}

trc_v6e_gradient_orientation_entropy = {
    "common": {
        "logbase": f"{user.bucket}/logs/trc-v6e-gradient-orientation-entropy",
        "prompt_fn": "imagenet_animals",
        "filter_field": "gradient_orientation_entropy",
    },
    "pg": _pg_trc,
}
