# MOSAIC Release

This folder contains the packaged training, evaluation, and inference code for the released MOSAIC model, together with the processed benchmark datasets, the local DNABERT-2 backbone files, and the best trained checkpoint renamed to `MOSAIC_model.ckpt`.

## Contents

- `train_MOSAIC.py`: training entry point for the released MOSAIC configuration.
- `evaluate_MOSAIC.py`: evaluation script that reports overall, per-dataset, and per-type metrics.
- `infer_MOSAIC.py`: inference script for single-sequence or batch CSV prediction.
- `configs/MOSAIC.yaml`: released model configuration without seed-specific naming.
- `checkpoints/MOSAIC_model.ckpt`: best trained MOSAIC checkpoint.
- `hf_models/DNABERT-2-117M/`: local DNABERT-2 backbone files required by MOSAIC.
- `datasets/`: processed train/test benchmark splits used by the released code.
- `data/MOSAIC_data.py`: released data module for the processed benchmark.
- `models/MOSAIC.py`: released MOSAIC architecture definition.
- `utils/MOSAIC_utils.py`: packaged helper functions for reproducibility and class weighting.
- `requirements.txt`: Python package dependencies used for this release.

## Environment

Python 3.10+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU usage, install the PyTorch build that matches your CUDA environment if the default `torch==2.6.0` wheel is not appropriate.

## Training

Run:

```bash
python train_MOSAIC.py --config configs/MOSAIC.yaml
```

The released training code no longer exposes a seed-specific file name. Internally, a fixed seed of `42` is used for deterministic splitting and dataloader reproducibility.

Training outputs are written under `outputs/train/`.

## Evaluation

Run:

```bash
python evaluate_MOSAIC.py --config configs/MOSAIC.yaml --checkpoint checkpoints/MOSAIC_model.ckpt
```

The JSON report is written to `outputs/eval/`.

## Inference

### Single sequence

```bash
python infer_MOSAIC.py --sequence ATCGATCGATCGATCGATCGATCGATCGATCGATCGA --species A.thaliana --task 6mA
```

### Batch inference

Prepare a CSV with columns:

```text
sequence,species,task
ATCGATCGATCGATCGATCGATCGATCGATCGATCGA,A.thaliana,6mA
GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCG,H.sapiens,5hmC
```

Run:

```bash
python infer_MOSAIC.py --input-csv examples/inference_example.csv --output-csv outputs/inference_results.csv
```

The inference script returns the binary prediction probabilities and the routed Top-K experts with their weights.

## Processed datasets

The packaged benchmark keeps only the processed `train.csv` and `test.csv` splits for each dataset, together with `datasets/summary.csv`. Auxiliary baseline feature files used elsewhere in the development repo are intentionally excluded from this release package.

## Notes

- Supported task values are `4mC`, `5hmC`, and `6mA`.
- Species names must match the released benchmark vocabulary. The inference script validates them against the packaged datasets.
- The released checkpoint was renamed from the internal training artifact to `MOSAIC_model.ckpt` for clarity.
