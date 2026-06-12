# Simulated Imbalance Stress-Test Bundle

This bundle contains the 1:5 and 1:10 simulated imbalance test sets used for the revised MOSAIC benchmark stress test.

The original balanced benchmark splits under `datasets/` are unchanged. These stress-test sets retain the original test positives and add same-species central-base genome-candidate sequences treated as unreported for methylation.

## Contents

- `simulated_imbalance_tests.zip`: packaged CSV test files.
- `manifest.csv`: per-dataset counts and archive paths.
- `summary.csv`: ratio-level sample counts.
- `validation_audit_public_terms.csv`: validation checks for sequence length, central base, labels, and duplicate handling.
- `coverage_audit_public_provenance.csv`: source and coverage audit for the constructed candidate pools.

Inside `simulated_imbalance_tests.zip`, the test files are arranged as:

```text
data/
  ratio_1to5/
    4mC|5hmC|6mA/<dataset>/test.csv
  ratio_1to10/
    4mC|5hmC|6mA/<dataset>/test.csv
```

Each `test.csv` uses the columns:

```text
id,sequence,label,dataset,source_level,source_name,source_id
```

`label=1` denotes the original benchmark test positives. `label=0` denotes same-species candidate sequences unreported for methylation under the construction audit.
