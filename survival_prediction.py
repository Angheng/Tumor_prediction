from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


DEFAULT_DATA_DIR = Path("2.curated_data")
DEFAULT_MODE = "validation"
RANDOM_STATE = 42
TEST_SIZE = 0.2
NAN_FILL_VALUE = -1

FEATURES = [
    "ENSG00000258235.1",
    "ENSG00000251751.1",
    "ENSG00000171209.3",
    "ENSG00000173825.7",
    "ENSG00000228530.1",
    "ENSG00000272945.1",
    "ENSG00000105607.13",
    "ENSG00000263503.1",
    "ENSG00000253227.2",
    "ENSG00000078369.18",
    "ENSG00000262539.1",
    "ENSG00000110911.16",
]


def _load_subtype_dataset(subtype: str, data_dir: Path, dataset_split: str) -> tuple[pd.DataFrame, pd.Series]:
    expr_path = data_dir / f"{subtype}_{dataset_split}_rna_expression_tumor.tsv"
    survival_path = data_dir / f"{subtype}_{dataset_split}_overall_survival.tsv"

    expr = pd.read_csv(expr_path, sep="\t").set_index("idx")
    survival = pd.read_csv(survival_path, sep="\t").set_index("case_id")

    missing_features = sorted(set(FEATURES) - set(expr.index))
    if missing_features:
        raise ValueError(f"{expr_path} is missing selected features: {missing_features}")

    common_samples = survival.index.intersection(expr.columns)
    if len(common_samples) != len(survival):
        missing_samples = sorted(set(survival.index) - set(expr.columns))
        raise ValueError(f"{expr_path} is missing survival samples: {missing_samples}")

    x = expr.loc[FEATURES, survival.index].T
    x.index = [f"{subtype}_{sample}" for sample in x.index]
    y = survival["OS_event"].astype(int)
    y.index = x.index
    return x, y


def load_dataset(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    dataset_split: str = "trainingset",
) -> tuple[pd.DataFrame, pd.Series]:
    data_dir = Path(data_dir)
    frames: list[pd.DataFrame] = []
    labels: list[pd.Series] = []

    for subtype in ("LUAD", "LSCC"):
        x_subtype, y_subtype = _load_subtype_dataset(subtype, data_dir, dataset_split)
        frames.append(x_subtype)
        labels.append(y_subtype)

    x = pd.concat(frames, axis=0).fillna(NAN_FILL_VALUE)
    y = pd.concat(labels, axis=0).astype(int)
    return x, y


def build_model() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    )


def _evaluate(y_true: pd.Series, y_pred) -> dict[str, float | str]:
    return {
        "Project": "Survival Prediction",
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Accuracy": accuracy_score(y_true, y_pred),
    }


def _train_and_evaluate(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float | str]:
    model = build_model()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return _evaluate(y_test, y_pred)


def run_validation(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict[str, float | str]:
    x, y = load_dataset(data_dir, dataset_split="trainingset")
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return _train_and_evaluate(x_train, y_train, x_test, y_test)


def run_test(data_dir: str | Path = DEFAULT_DATA_DIR, test_data_path: str | Path | None = None) -> dict[str, float | str]:
    if test_data_path is None:
        raise ValueError("test_data_path is required when mode is 'test'.")

    x_train, y_train = load_dataset(data_dir, dataset_split="trainingset")
    x_test, y_test = load_dataset(test_data_path, dataset_split="testset")
    return _train_and_evaluate(x_train, y_train, x_test, y_test)


def run(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    mode: str = DEFAULT_MODE,
    test_data_path: str | Path | None = None,
) -> dict[str, float | str]:
    if mode == "validation":
        return run_validation(data_dir)

    if mode == "test":
        return run_test(data_dir, test_data_path)

    raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train/evaluate the survival prediction classifier.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, type=Path, help="Directory containing curated TSV datasets.")
    parser.add_argument("--mode", choices=("validation", "test"), default=DEFAULT_MODE, help="Evaluation mode.")
    parser.add_argument(
        "--test-data-path",
        "--test-data-dir",
        dest="test_data_path",
        default=None,
        type=Path,
        help="Directory containing testset TSV datasets. Required in test mode.",
    )
    args = parser.parse_args()

    if args.mode == "test" and args.test_data_path is None:
        parser.error("--test-data-path is required when --mode test.")

    print(pd.DataFrame([run(args.data_dir, args.mode, args.test_data_path)]).to_string(index=False))
