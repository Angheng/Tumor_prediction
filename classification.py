from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import luad_vs_lscc
import survival_prediction
import tumor_vs_normal


DEFAULT_DATA_DIR = Path("2.curated_data")
DEFAULT_MODE = "validation"
DEFAULT_RESULT_FORMAT = "csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all three project classifiers and save metric results.")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        type=Path,
        help="Directory containing trainingset TSV datasets.",
    )
    parser.add_argument(
        "--mode",
        choices=("validation", "test"),
        default=DEFAULT_MODE,
        help="validation splits data-dir 8:2; test trains on all data-dir and evaluates test-data-path.",
    )
    parser.add_argument(
        "--test-data-path",
        "--test-data-dir",
        dest="test_data_path",
        default=None,
        type=Path,
        help="Directory containing testset TSV datasets. Required when --mode test.",
    )
    parser.add_argument(
        "--result-format",
        choices=("csv", "json"),
        default=DEFAULT_RESULT_FORMAT,
        help="Output file format for the result table.",
    )
    parser.add_argument(
        "--result-path",
        type=Path,
        default=None,
        help="Output file path. Defaults to result.<result-format>.",
    )
    args = parser.parse_args()
    if args.mode == "test" and args.test_data_path is None:
        parser.error("--test-data-path is required when --mode test.")
    return args


def save_results(result_df: pd.DataFrame, result_path: Path, result_format: str) -> None:
    if result_format == "csv":
        result_df.to_csv(result_path, index=False)
        return

    if result_format == "json":
        result_df.to_json(result_path, orient="records", indent=2)
        return

    raise ValueError(f"Unsupported result format: {result_format}")


def main(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    mode: str = DEFAULT_MODE,
    test_data_path: str | Path | None = None,
    result_format: str = DEFAULT_RESULT_FORMAT,
    result_path: str | Path | None = None,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    test_data_path = Path(test_data_path) if test_data_path is not None else None
    result_path = Path(result_path) if result_path is not None else Path(f"result.{result_format}")

    results = [
        tumor_vs_normal.run(data_dir, mode, test_data_path),
        luad_vs_lscc.run(data_dir, mode, test_data_path),
        survival_prediction.run(data_dir, mode, test_data_path),
    ]

    result_df = pd.DataFrame(
        results,
        columns=["Project", "Precision", "Recall", "F1-Score", "Accuracy"],
    )
    save_results(result_df, result_path, result_format)
    print(result_df.to_string(index=False))
    print(f"\nSaved results to {result_path}")
    return result_df


if __name__ == "__main__":
    args = parse_args()
    main(args.data_dir, args.mode, args.test_data_path, args.result_format, args.result_path)
