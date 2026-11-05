from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"


def dataset_info(dataset, meta=None):
    """Return metadata associated with supported datasets."""
    info = {
        "CelebA": {
            "orig_res": (178, 218),
            "img_dir": str(DATA_ROOT / "CelebA" / "img_align_celeba") + "/",
            "target_res": (224, 224),
            "transform": "base",
        },
    }
    if meta is None:
        return info[dataset]
    return info[dataset][meta]
