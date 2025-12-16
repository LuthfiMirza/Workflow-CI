"""
Modelling script untuk MLflow Project.
Menerima argumen CLI (data_path, n_estimators) dan melog manual ke MLflow.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


# Tahap parsing argumen
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model training inside MLflow Project")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path ke data preprocessed (CSV).",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=50,
        help="Jumlah trees untuk RandomForest.",
    )
    return parser.parse_args()


# Tahap training dan logging
def run_training(data_path: str, n_estimators: int) -> None:
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        raise FileNotFoundError(f"File data tidak ditemukan: {data_path_obj}")

    df = pd.read_csv(data_path_obj)
    target_col = "label"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Dataset kecil: test_size adaptif dan stratify hanya jika tiap kelas punya ≥2 sampel
    n_classes = y.nunique()
    min_test_size = max(n_classes, 1) / len(df)
    test_size = max(0.2, min(0.4, min_test_size))
    stratify_param = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify_param,
    )

    existing_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if existing_tracking_uri:
        tracking_uri = existing_tracking_uri
    else:
        mlruns_dir_env = os.environ.get("MLRUNS_DIR")
        mlruns_dir = Path(mlruns_dir_env) if mlruns_dir_env else Path.cwd() / "mlruns"
        mlruns_dir.mkdir(parents=True, exist_ok=True)
        tracking_uri = "file://" + str(mlruns_dir.resolve())

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("workflow-ci-training")

    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=42, n_jobs=-1
    )

    # Hapus MLFLOW_RUN_ID dari env untuk menghindari bentrok run parent dari `mlflow run`
    os.environ.pop("MLFLOW_RUN_ID", None)
    with mlflow.start_run(run_name="ci-random-forest") as run:
        mlflow.log_param("n_estimators", n_estimators)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Run ID: {run.info.run_id}")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


def main() -> None:
    args = parse_args()
    run_training(args.data_path, args.n_estimators)


if __name__ == "__main__":
    main()
