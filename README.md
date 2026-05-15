# Project Classification Scripts

These Python files train and evaluate the three classification tasks from `7_Project.pdf` using TSV datasets.

## Files

- `classification.py`: runs all three tasks and saves one metrics table.
- `tumor_vs_normal.py`: trains the Tumor vs Normal classifier.
- `luad_vs_lscc.py`: trains the LUAD vs LSCC classifier.
- `survival_prediction.py`: trains the Survival vs Death classifier.

## Requirements

Install the Python packages below before running the scripts:

```bash
python -m pip install pandas numpy scikit-learn xgboost
```

## Dataset File Names

Validation mode reads training files whose names contain `trainingset`, for example:

```text
LUAD_trainingset_rna_expression_tumor.tsv
LUAD_trainingset_rna_expression_nat.tsv
LUAD_trainingset_overall_survival.tsv
```

Test mode expects the same file format in a separate folder, but with `trainingset` changed to `testset`, for example:

```text
LUAD_testset_rna_expression_tumor.tsv
LUAD_testset_rna_expression_nat.tsv
LUAD_testset_overall_survival.tsv
```

The same naming rule is used for `LSCC`.

## Validation Mode

Validation mode is the default current process. It reads all data from `--data-dir`, makes an 8:2 train/test split with `random_state=42`, trains on the training split, and evaluates on the validation split.

Default run:

```bash
python classification.py
```

This reads training datasets from `2.curated_data/` and saves `result.csv`.

Use a different dataset directory:

```bash
python classification.py --mode validation --data-dir 2.curated_data
```

Save CSV results to a custom path:

```bash
python classification.py --mode validation --data-dir 2.curated_data --result-format csv --result-path result.csv
```

Save JSON results:

```bash
python classification.py --mode validation --data-dir 2.curated_data --result-format json --result-path result.json
```

## Test Mode

Test mode trains on all samples in `--data-dir`. It does not split the training dataset. It evaluates on files in `--test-data-path`.

```bash
python classification.py --mode test --data-dir 2.curated_data --test-data-path path/to/test_data --result-format csv --result-path result.csv
```

JSON output works in test mode too:

```bash
python classification.py --mode test --data-dir 2.curated_data --test-data-path path/to/test_data --result-format json --result-path result.json
```

## Output Columns

The result table contains:

- `Project`
- `Precision`
- `Recall`
- `F1-Score`
- `Accuracy`

## Run One Task

Each task script can also be run directly:

```bash
python tumor_vs_normal.py --mode validation --data-dir 2.curated_data
python luad_vs_lscc.py --mode validation --data-dir 2.curated_data
python survival_prediction.py --mode validation --data-dir 2.curated_data
```

Direct test-mode examples:

```bash
python tumor_vs_normal.py --mode test --data-dir 2.curated_data --test-data-path path/to/test_data
python luad_vs_lscc.py --mode test --data-dir 2.curated_data --test-data-path path/to/test_data
python survival_prediction.py --mode test --data-dir 2.curated_data --test-data-path path/to/test_data
```

## Modeling Details

All scripts use:

- Selected RNA features from `dataset_analy.ipynb`
- Validation mode: `train_test_split` with `test_size=0.2`, `random_state=42`, and stratified labels
- Test mode: train on all `trainingset` files and evaluate on `testset` files
- Missing-value fill value of `-1`
- `XGBClassifier`
