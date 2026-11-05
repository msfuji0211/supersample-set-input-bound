# Supersample Set-Input Bound Experiments (CelebA / ViT)

This repository contains a **minimal, self-contained** release for reproducing the
paper's numerical bound evaluation on **CelebA** with **ViT**.

The pipeline estimates single-index local information terms for two set-input
objectives:

- `worst`: finite bin/group/class multicalibration moment diagnostic
- `cvar`: CVaR of absolute calibration residual `|p - y|` (default `α=0.2`)

## Repository layout

```text
.
├── Dataset.py
├── Model.py
├── utils.py
├── configs/
│   ├── datasets.py
│   └── hyperparameters.py
├── dataloaders/
│   ├── CelebA.py
│   └── utils/
├── models/
│   └── ViT.py
├── mcb_algorithms/
│   └── HJZ/calibration.py
├── scripts/
│   ├── run_supersample_retrain_worst_train.py
│   ├── run_supersample_retrain_worst_eval.py
│   ├── run_supersample_retrain_worst_cvar_eval.py
│   ├── rebuild_supersample_retrain_summary.py
│   ├── plot_supersample_retrain_bound.py
│   └── supersample_retrain_worst_common.py
├── data/CelebA/          # place CelebA here
└── results/              # created at runtime
```

## Installation

```bash
conda create -n supersample-bound python=3.10 -y
conda activate supersample-bound
pip install -r requirements.txt

cd /path/to/supersample-set-input-bound
export PYTHONPATH=$PWD:$PYTHONPATH
```

A CUDA-enabled PyTorch build is strongly recommended for ViT retraining.

## Data setup

Place CelebA under:

```text
data/CelebA/
├── list_attr_celeba.csv
└── img_align_celeba/
    └── *.jpg
```

If files are missing, the loader may attempt an automatic download. Manual
placement is more reliable for large image archives.

## Reproduce the paper experiment

Paper settings:

- dataset / model: `CelebA` / `ViT`
- supersample seeds: `10,15,20`
- model seeds: `0,1,2,3,4,5,6`
- sample-size percents: `5,10,25,50,75,100`
  (`N ∈ {8104,16208,40520,81040,121560,162080}`)
- selected coordinates: `m=10`
- objectives: multicalibration (`worst`) and CVaR (`α=0.2`)

### 1. Train branch models

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_supersample_retrain_worst_train.py \
  --datasets CelebA \
  --models ViT \
  --supersample-seeds 10,15,20 \
  --model-seeds 0,1,2,3,4,5,6 \
  --sample-size-percents 5,10,25,50,75,100 \
  --n-selected-indices 10 \
  --output-dir results/supersample_retrain_worst_train \
  --gpu 0 \
  --num-workers 4
```

### 2. Evaluate worst + CVaR

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_supersample_retrain_worst_cvar_eval.py \
  --datasets CelebA \
  --models ViT \
  --supersample-seeds 10,15,20 \
  --model-seeds 0,1,2,3,4,5,6 \
  --sample-size-percents 5,10,25,50,75,100 \
  --cvar-alpha 0.2 \
  --input-dir results/supersample_retrain_worst_train \
  --output-dir-base results/supersample_retrain_phi_eval
```

Outputs:

```text
results/supersample_retrain_phi_eval/worst/
results/supersample_retrain_phi_eval/cvar_a0p2/
```

### 3. Plot bound curves

```bash
python scripts/plot_supersample_retrain_bound.py \
  --summary-csv results/supersample_retrain_phi_eval/worst/summary.csv \
  --cvar-summary-csv results/supersample_retrain_phi_eval/cvar_a0p2/summary.csv
```

### Optional: rebuild summary from partial `local_rows_*.csv`

```bash
python scripts/rebuild_supersample_retrain_summary.py \
  --eval-dir results/supersample_retrain_phi_eval/worst \
  --output-csv results/supersample_retrain_phi_eval/worst/summary.csv \
  --models ViT
```

## Smoke test (small / fast)

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_supersample_retrain_worst_train.py \
  --datasets CelebA --models ViT \
  --supersample-seeds 10 --model-seeds 0 \
  --sample-size-percents 5 --n-selected-indices 1 \
  --output-dir results/smoke_train --gpu 0 --num-workers 2

CUDA_VISIBLE_DEVICES=0 python scripts/run_supersample_retrain_worst_cvar_eval.py \
  --datasets CelebA --models ViT \
  --supersample-seeds 10 --model-seeds 0 \
  --sample-size-percents 5 --cvar-alpha 0.2 \
  --input-dir results/smoke_train \
  --output-dir-base results/smoke_eval
```

## Notes

- Keep `--sample-size-percents` / `--sample-sizes` identical between train and eval.
- Evaluation reads `metadata/*/supersample_seed=*.npz` (`selected_indices`),
  not the `models/` folder listing.
- First ViT run downloads `google/vit-base-patch16-224-in21k` from Hugging Face.

## Attribution

This release adapts infrastructure from Hansen et al.'s empirical multicalibration
codebase (MIT License). See `LICENSE` and `NOTICE`.
