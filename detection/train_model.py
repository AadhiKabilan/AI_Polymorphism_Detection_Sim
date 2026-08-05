import os
import json
import joblib
import argparse
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Safety: Works on CSV features only.


def _prepare_xy(df: pd.DataFrame):
    # Encode labels: benign=0, malware_sim=1
    y = (df["label"].astype(str) == "malware_sim").astype(int).values
    feature_cols = [
        "num_sections",
        "avg_section_entropy",
        "total_section_size",
        "imported_api_count",
        "strings_count",
        "junk_bytes_count",
        "resource_size",
    ]
    X = df[feature_cols].values.astype(float)
    return X, y, feature_cols


def _apply_heuristic(row: pd.Series, junk_threshold: int = 2500, entropy_threshold: float = 6.0) -> int:
    return int((row["junk_bytes_count"] > junk_threshold) and (row["avg_section_entropy"] > entropy_threshold))


def train_model(csv_path: str, out_dir: str, do_grid: bool = False) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    X, y, feature_cols = _prepare_xy(df)

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)

    if do_grid:
        params = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
        }
        grid = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1), params, cv=3, n_jobs=-1, scoring="f1")
        grid.fit(X_train, y_train)
        clf = grid.best_estimator_

    clf.fit(X_train, y_train)

    prob_test = clf.predict_proba(X_test)[:, 1]
    pred_ml = (prob_test >= 0.5).astype(int)

    # Heuristic layer on test
    heuristic_flags = df_test.apply(_apply_heuristic, axis=1).values

    # Simple ensemble: if ML or heuristic says 1, final is 1
    pred_final = np.maximum(pred_ml, heuristic_flags)

    acc = accuracy_score(y_test, pred_final)
    prec = precision_score(y_test, pred_final, zero_division=0)
    rec = recall_score(y_test, pred_final, zero_division=0)
    cm = confusion_matrix(y_test, pred_final).tolist()

    model_path = os.path.join(out_dir, "random_forest.pkl")
    joblib.dump({
        "model": clf,
        "feature_cols": feature_cols,
        "heuristic": {"junk_threshold": 2500, "entropy_threshold": 6.0},
    }, model_path)

    importances = getattr(clf, "feature_importances_", None)
    if importances is not None:
        importances = importances.tolist()

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm,
        "feature_cols": feature_cols,
        "feature_importances": importances,
        "samples": {"train": int(len(X_train)), "test": int(len(X_test))},
    }

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    return {"model_path": model_path, "metrics_path": metrics_path, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument("--out", default="detection/model", help="Output dir for model and metrics")
    parser.add_argument("--grid", action="store_true", help="Enable small GridSearchCV")
    args = parser.parse_args()
    train_model(args.data, args.out, do_grid=args.grid)


if __name__ == "__main__":
    # Example minimal usage (requires dataset built):
    #   python detection/train_model.py --data dataset/combined_features.csv --out detection/model
    try:
        main()
    except SystemExit:
        # Allow import/run without CLI
        pass
