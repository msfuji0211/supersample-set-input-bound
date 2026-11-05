"""Hyperparameters used by the released CelebA / ViT supersample experiments."""

hyperparameters = {
    "ViT": {
        "CelebA": {
            cf: {
                "epochs": 10,
                "batch_size": 64,
                "lr_schedule": {0: 1e-4},
                "weight_decay": 0.01,
                "optim": "adam",
                "momentum": 0,
            }
            for cf in [0, 0.2, 0.4]
        },
    },
}


def get_hyperparameters(model, dataset, calib_frac):
    return hyperparameters[model][dataset][calib_frac]
