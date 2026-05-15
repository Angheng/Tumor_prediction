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

## Experimental Setting

The experiment uses the selected RNA features from `dataset_analy.ipynb` for all three project tasks.

Experimental environment:

- OS: Ubuntu 24.04.3 LTS
- Kernel/platform: Linux 6.6.87.2 Microsoft WSL2, x86_64
- CPU: AMD Ryzen Threadripper 2990WX 32-Core Processor
- CPU cores/threads: 32 cores, 64 threads
- Memory: 62 GiB RAM
- Python: 3.13.5
- pandas: 2.2.3
- numpy: 2.1.3
- scikit-learn: 1.6.1
- xgboost: 3.2.0

Task definitions:

- Tumor vs Normal: LUAD and LSCC tumor samples are labeled `1`; LUAD and LSCC NAT samples are labeled `0`.
- LUAD vs LSCC: LUAD tumor samples are labeled `0`; LSCC tumor samples are labeled `1`.
- Survival Prediction: tumor samples are labeled by `OS_event` from the survival table.

Data processing:

- Input files are TSV matrices where rows are gene features and columns are patient/sample IDs.
- Expression matrices are transposed before modeling so each row is one sample and each column is one selected feature.
- Survival labels are aligned to expression columns by `case_id`.
- Missing values are filled with `-1`.

Validation setup:

- The current default experiment is `validation` mode.
- The full training dataset from `--data-dir` is split into train and validation sets using an 8:2 split.
- The split uses `random_state=42` and `stratify=y`.

Test setup:

- In `test` mode, every sample in `--data-dir` is used for training.
- The external test folder is provided by `--test-data-path`.
- Test files must use the same file structure as the training files, but with `testset` instead of `trainingset`.

Model setup:

- Model: `XGBClassifier`
- Objective: `binary:logistic`
- Evaluation metric during training: `logloss`
- `n_estimators=100`
- `max_depth=3`
- `learning_rate=0.1`
- `subsample=0.9`
- `colsample_bytree=0.9`
- `random_state=42`

Reported metrics:

- Weighted Precision
- Weighted Recall
- Weighted F1-Score
- Accuracy

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
