from . import user
from .base import base

# DDPO policy-gradient config tuned for TRC v6e-8 (8 chips, single host)
# sample_batch_size=4 per device x 8 devices = 32 images/batch
# train_batch_size=1 per device x 8 devices x train_accumulation_steps=4 = 32 effective

_pg_trc = {
    "sample_batch_size": 4,
    "num_sample_batches_per_epoch": 2,
    "train_batch_size": 1,
    "train_accumulation_steps": 4,
    "num_train_epochs": 100,
    "save_freq": 10,
}

trc_v6e_compressed = {
    "common": {
        "logbase": f"{user.bucket}/logs/trc-v6e-compressed",
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
