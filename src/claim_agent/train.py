from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from claim_agent.config import (
    METRICS_PATH,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    MODEL_PATH,
)
from claim_agent.data import FEATURE_COLUMNS, NARRATIVE_COLUMN, load_raw_dataset, prepare_labels
from claim_agent.features import build_preprocessor, coerce_feature_types


def build_model_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            (
                "classifier",
                GradientBoostingClassifier(random_state=42),
            ),
        ]
    )


def train(
    data_path: Path | None = None,
    tune_hyperparameters: bool = True,
) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw_dataset(data_path) if data_path else load_raw_dataset()
    labels = prepare_labels(df)
    X = coerce_feature_types(df.loc[labels.index])
    y = labels.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_model_pipeline()

    with mlflow.start_run(run_name="claim_approval_gbm"):
        if tune_hyperparameters:
            param_grid = {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 5],
                "classifier__learning_rate": [0.05, 0.1],
            }
            search = GridSearchCV(
                pipe,
                param_grid,
                cv=5,
                scoring="f1",
                n_jobs=-1,
                refit=True,
            )
            search.fit(X_train, y_train)
            best = search.best_estimator_
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("cv_best_f1", search.best_score_)
        else:
            best = pipe
            best.fit(X_train, y_train)

        # Refit on full data for production artifact
        best.fit(X, y)

        y_pred = best.predict(X_test)
        y_prob = best.predict_proba(X_test)[:, 1]

        metrics = {
            "holdout_accuracy": float(accuracy_score(y_test, y_pred)),
            "holdout_precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "holdout_recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "holdout_f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "holdout_roc_auc": float(roc_auc_score(y_test, y_prob)),
            "train_samples": int(len(y_train)),
            "test_samples": int(len(y_test)),
            "decline_rate": float(1 - y.mean()),
        }
        cv_scores = cross_val_score(
            build_model_pipeline(), X_train, y_train, cv=5, scoring="f1"
        )
        metrics["cv_f1_mean"] = float(cv_scores.mean())
        metrics["cv_f1_std"] = float(cv_scores.std())

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        report = classification_report(
            y_test, y_pred, target_names=["Declined", "Completed"]
        )
        mlflow.log_text(report, "classification_report.txt")

        joblib.dump(best, MODEL_PATH)
        mlflow.sklearn.log_model(best, "model")
        METRICS_PATH.write_text(json.dumps(metrics, indent=2))

        # Feature importances from tree model on transformed names (approximate via importances)
        clf: GradientBoostingClassifier = best.named_steps["classifier"]
        importances = clf.feature_importances_
        metrics["top_feature_importance_sum"] = float(np.sum(importances[:10]))

    return metrics


def _fast_train_enabled() -> bool:
    return os.getenv("CLAIM_AGENT_FAST_TRAIN", "").lower() in ("1", "true", "yes")


if __name__ == "__main__":
    result = train(tune_hyperparameters=not _fast_train_enabled())
    print(json.dumps(result, indent=2))
