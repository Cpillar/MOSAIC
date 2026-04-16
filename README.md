# MOSAIC

MOSAIC is a prompt-aware mixture-of-experts framework for cross-species and cross-modification DNA methylation prediction. It combines a shared DNABERT-2 backbone with sparse expert routing conditioned on methylation type and species prompts, allowing the model to preserve strong predictive performance while retaining biologically structured transfer across heterogeneous datasets.

Many methylation benchmarks are highly imbalanced: a small number of head datasets dominate training, while many low-resource datasets remain much harder to model well. In this setting, a method should not only improve average predictive performance, but also avoid concentrating those gains on already favorable tasks. MOSAIC is designed for that setting. In addition to benchmark-level accuracy, it targets fairer task-level behavior across datasets with different methylation types, species backgrounds, and sample sizes.

![MOSAIC architecture](docs/mosaic_architecture.png)

## Live web server

For users who only need online inference, MOSAIC is also available through a public web interface:

- [MOSAIC Web Server](https://ycclab.cuhk.edu.cn/MOSAIC/index.php)

This release package focuses on the released model, processed datasets, and local training / evaluation / inference scripts.

## Release contents

- `train_MOSAIC.py`: training entry point for the released MOSAIC configuration.
- `evaluate_MOSAIC.py`: evaluation script for overall, per-dataset, and per-type metrics.
- `infer_MOSAIC.py`: inference script for single-sequence or batch CSV prediction.
- `configs/MOSAIC.yaml`: released model configuration.
- `checkpoints/MOSAIC_model.ckpt`: best trained MOSAIC checkpoint.
- `hf_models/DNABERT-2-117M/`: local DNABERT-2 backbone files required by MOSAIC.
- `datasets/`: processed benchmark train/test splits and `summary.csv`.
- `external_ood_zero_shot/`: three released external OOD datasets and a prompt-sweep inference example.
- `data/MOSAIC_data.py`: released data module.
- `models/MOSAIC.py`: released MOSAIC architecture definition.
- `utils/MOSAIC_utils.py`: helper functions used by the release code.
- `requirements.txt`: Python package dependencies for the released pipeline.

## Environment

Python `3.10+` is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

If your environment requires a different CUDA-specific PyTorch build, install the matching `torch` wheel manually before running the scripts.

## Training

```bash
python train_MOSAIC.py --config configs/MOSAIC.yaml
```

Training outputs are written to `outputs/train/`.

## Evaluation

```bash
python evaluate_MOSAIC.py --config configs/MOSAIC.yaml --checkpoint checkpoints/MOSAIC_model.ckpt
```

Evaluation outputs are written to `outputs/eval/`.

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

Then run:

```bash
python infer_MOSAIC.py --input-csv examples/inference_example.csv --output-csv outputs/inference_results.csv
```

The inference script reports methylation probabilities together with the routed Top-K experts and their routing weights.

## Processed datasets

This release keeps only the processed `train.csv` and `test.csv` splits for each dataset, together with `datasets/summary.csv`. Development-time auxiliary baseline features are intentionally excluded from the release package.

## External OOD bundle

The directory `external_ood_zero_shot/` contains the three paper-facing external datasets used for zero-shot OOD evaluation:

- `5mC / H.sapiens`
- `6mA / O.sativa`
- `4mC / E.coli`

It also includes a prompt-sweep inference example that can be run directly with `infer_MOSAIC.py`. The paper-level zero-shot OOD metrics are not bundled in the release branch and can be regenerated locally from the released checkpoint and the included external datasets.

## Notes

- Supported methylation tasks are `4mC`, `5hmC`, and `6mA`.
- Species names must match the released benchmark vocabulary.
- The released checkpoint was renamed from the internal training artifact to `MOSAIC_model.ckpt` for clarity.
