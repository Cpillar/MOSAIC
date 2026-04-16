# External OOD Zero-Shot Bundle

This bundle contains the three paper-facing external datasets used for MOSAIC zero-shot out-of-distribution (OOD) evaluation, together with the corresponding MOSAIC zero-shot metrics and the baseline comparison summaries.

## Included external datasets

- `datasets/5mC/H.sapiens_promoter_BERT5mC/`
  - same-species, unseen-type setting
  - official independent-test partition only
- `datasets/6mA/O.sativa_Rice_Chen/`
  - same-type, unseen-species setting
- `datasets/4mC/E.coli_Li2020/`
  - same-type, unseen-species setting

Each dataset directory contains:

- `test.csv`: normalized MOSAIC-compatible external test file with columns `id,sequence,label`
- `metadata.json`: source paper, source repository, OOD axis, and dataset-level counts

The file `manifest.csv` records the released three-dataset inventory.

## Included zero-shot results

- `results/mosaic/<dataset_name>/metrics.json`
  - MOSAIC zero-shot prompt-sweep metrics for the corresponding external dataset
  - includes per-prompt metrics and aggregate metrics
- `results/compare_single_zero_shot/other_vs_mosaic_summary.csv`
  - paper-facing comparison table between MOSAIC and the public single-dataset baselines
- `results/compare_single_zero_shot/other_vs_mosaic_summary.json`
  - JSON version of the same summary
- `results/compare_single_zero_shot/per_model_summary.csv`
  - per-baseline external results under the same evaluation protocol

## How to run OOD-style prediction

The released `infer_MOSAIC.py` script accepts only species and task values from the released benchmark vocabulary. Therefore, OOD prediction is performed as a prompt sweep over seen prompts while keeping the external sequence fixed.

The example file:

- `examples/ood_prompt_sweep_example.csv`

contains three prompt-sweep groups:

- the same external human 5mC sequence evaluated with `H.sapiens` and tasks `{4mC, 5hmC, 6mA}`
- the same external rice 6mA sequence evaluated with task `6mA` and seen species prompts `{A.thaliana, F.vesca, C.equisetifolia, R.chinensis}`
- the same external *E.coli* 4mC sequence evaluated with task `4mC` and seen species prompts `{A.thaliana, C.elegans, D.melanogaster, C.equisetifolia, F.vesca, S.cerevisiae, Tolypocladium}`

From the repository root, run:

```bash
python infer_MOSAIC.py --input-csv external_ood_zero_shot/examples/ood_prompt_sweep_example.csv --output-csv outputs/ood_prompt_sweep_example_predictions.csv
```

The output CSV will contain:

- methylation probabilities
- predicted class
- routed Top-K experts
- routing weights

To reproduce the full paper-level OOD benchmark, use the external `test.csv` files in this bundle and aggregate the prompt-level outputs according to the paper protocol described in `results/mosaic/*/metrics.json`.

## Sources

- Human 5mC benchmark: BERT-5mC
- Rice 6mA benchmark: i6mA-Pred / SNNRice6mA public benchmark mirror
- *E.coli* 4mC benchmark: MSNet-4mC / Li_2020 test benchmark
