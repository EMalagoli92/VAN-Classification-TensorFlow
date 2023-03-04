from typing import Dict

MODELS_CONFIG: Dict[str, dict] = {
    "van_b0": {
        "spec": {
            "embed_dims": [32, 64, 160, 256],
            "mlp_ratios": [8, 8, 4, 4],
            "depths": [3, 3, 5, 2],
            "name": "van_b0",
        },
        "pretrained_img_resolution": 224,
    },
    "van_b1": {
        "spec": {
            "embed_dims": [64, 128, 320, 512],
            "mlp_ratios": [8, 8, 4, 4],
            "depths": [2, 2, 4, 2],
            "name": "van_b1",
        },
        "pretrained_img_resolution": 224,
    },
    "van_b2": {
        "spec": {
            "embed_dims": [64, 128, 320, 512],
            "mlp_ratios": [8, 8, 4, 4],
            "depths": [3, 3, 12, 3],
            "name": "van_b2",
        },
        "pretrained_img_resolution": 224,
    },
    "van_b3": {
        "spec": {
            "embed_dims": [64, 128, 320, 512],
            "mlp_ratios": [8, 8, 4, 4],
            "depths": [3, 5, 27, 3],
            "name": "van_b3",
        },
        "pretrained_img_resolution": 224,
    },
    "van_b4": {
        "spec": {
            "embed_dims": [64, 128, 320, 512],
            "mlp_ratios": [8, 8, 4, 4],
            "depths": [3, 6, 40, 3],
            "name": "van_b4",
        },
        "pretrained_img_resolution": None,
    },
    "van_b5": {
        "spec": {
            "embed_dims": [96, 192, 480, 768],
            "mlp_ratios": [8, 8, 4, 4],
            "depths": [3, 3, 24, 3],
            "name": "van_b5",
        },
        "pretrained_img_resolution": None,
    },
    "van_b6": {
        "spec": {
            "embed_dims": [96, 192, 384, 768],
            "mlp_ratios": [8, 8, 4, 4],
            "depths": [6, 6, 90, 6],
            "name": "van_b6",
        },
        "pretrained_img_resolution": None,
    },
}


TF_WEIGHTS_URL: str = (
    "https://github.com/EMalagoli92/VAN-Classification-TensorFlow/releases/download"
)
