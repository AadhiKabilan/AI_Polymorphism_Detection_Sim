import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, GroupKFold, train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from detection.detector import (
    FEATURE_COLUMNS,
    DEFAULT_HEURISTIC,
    evaluate_heuristic,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_dataset(df: pd.DataFrame, feature_cols: List[str] = FEATURE_COLUMNS) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
    """Extract features matrix X, binary targets y, and group identifiers from DataFrame."""
    # Label encoding: malware_sim = 1, benign = 0
    if "label" not in df.columns:
        raise ValueError("Dataset CSV missing required 'label' column.")

    y = (df["label"].astype(str).str.strip().str.lower() == "malware_sim").astype(int).values

    # Clean numeric features
    for col in feature_cols:
        if col not in df.columns:
            logger.warning("Feature column '%s' missing from dataset, defaulting to 0", col)
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    X = df[feature_cols].values.astype(float)

    # Group identifier: origin_id defines the root variant family
    if "origin_id" in df.columns and df["origin_id"].notna().any():
        groups = df["origin_id"].fillna(df["id"] if "id" in df.columns else "unknown").astype(str)
    elif "id" in df.columns:
        groups = df["id"].astype(str)
    else:
        groups = pd.Series([f"group_{i}" for i in range(len(df))])

    return X, y, groups


def split_data_group_aware(
    X: np.ndarray,
    y: np.ndarray,
    groups: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform group-aware train/test split ensuring entire variant families are held out.

    Prevents data leakage where variants of the same parent artifact span both train and test partitions.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    # Verification: assert zero group intersection
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(f"Group leakage detected! Overlapping groups: {len(overlap)}")

    logger.info(
        "Group-aware split: %d train samples (%d families), %d test samples (%d families). Zero family overlap verified.",
        len(train_idx),
        len(train_groups),
        len(test_idx),
        len(test_groups),
    )

    return (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
        train_idx,
        test_idx,
    )


def train_model(
    csv_path: str,
    out_dir: str = "detection/model",
    split_method: str = "group",
    test_size: float = 0.2,
    random_state: int = 42,
    junk_threshold: int = DEFAULT_HEURISTIC["junk_threshold"],
    entropy_threshold: float = DEFAULT_HEURISTIC["entropy_threshold"],
) -> Dict[str, Any]:
    """Train Random Forest classifier with strict group-aware splitting and leakage-free pipeline.

    Args:
        csv_path: Path to dataset CSV.
        out_dir: Output directory for model and metrics.
        split_method: 'group' (leakage-free group split) or 'naive' (for benchmark comparison).
        test_size: Proportion of hold-out test set.
        random_state: Random seed for reproducibility.
        junk_threshold: Heuristic threshold for junk padding bytes.
        entropy_threshold: Heuristic threshold for average section entropy.

    Returns:
        Dictionary containing paths and evaluation metrics.
    """
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    X, y, groups = prepare_dataset(df, FEATURE_COLUMNS)

    if split_method == "group":
        X_train, X_test, y_train, y_test, train_idx, test_idx = split_data_group_aware(
            X, y, groups, test_size=test_size, random_state=random_state
        )
    else:
        logger.warning("Using NAIVE row-level split. This introduces variant lineage data leakage!")
        train_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=test_size, random_state=random_state, stratify=y
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    df_test = df.iloc[test_idx].copy()

    # Preprocessing and model pipeline: fitted strictly on training data
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    pipeline.fit(X_train, y_train)

    # 1. ML Model Evaluation
    prob_test = pipeline.predict_proba(X_test)[:, 1]
    pred_ml = (prob_test >= 0.5).astype(int)

    ml_acc = float(accuracy_score(y_test, pred_ml))
    ml_prec = float(precision_score(y_test, pred_ml, zero_division=0))
    ml_rec = float(recall_score(y_test, pred_ml, zero_division=0))
    ml_f1 = float(f1_score(y_test, pred_ml, zero_division=0))
    ml_auc = float(roc_auc_score(y_test, prob_test)) if len(np.unique(y_test)) > 1 else 0.0

    # 2. Heuristic Evaluation
    heuristic_flags = np.array([
        evaluate_heuristic(
            row.to_dict(),
            junk_threshold=junk_threshold,
            entropy_threshold=entropy_threshold,
        )
        for _, row in df_test.iterrows()
    ], dtype=int)

    h_acc = float(accuracy_score(y_test, heuristic_flags))
    h_prec = float(precision_score(y_test, heuristic_flags, zero_division=0))
    h_rec = float(recall_score(y_test, heuristic_flags, zero_division=0))
    h_f1 = float(f1_score(y_test, heuristic_flags, zero_division=0))

    # 3. Hybrid Ensemble Evaluation: max(ML, Heuristic)
    pred_final = np.maximum(pred_ml, heuristic_flags)
    ens_acc = float(accuracy_score(y_test, pred_final))
    ens_prec = float(precision_score(y_test, pred_final, zero_division=0))
    ens_rec = float(recall_score(y_test, pred_final, zero_division=0))
    ens_f1 = float(f1_score(y_test, pred_final, zero_division=0))
    cm = confusion_matrix(y_test, pred_final, labels=[0, 1]).tolist()
    report = classification_report(
        y_test,
        pred_final,
        labels=[0, 1],
        target_names=["benign", "malware_sim"],
        zero_division=0,
        output_dict=True,
    )

    # Cross-validation across groups on training set
    cv_mean, cv_std = 0.0, 0.0
    unique_train_groups = groups.iloc[train_idx].nunique()
    if unique_train_groups >= 3 and split_method == "group":
        n_splits = min(5, unique_train_groups)
        gkf = GroupKFold(n_splits=n_splits)
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            groups=groups.iloc[train_idx],
            cv=gkf,
            scoring="f1",
        )
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

    # Feature importances
    rf_clf: RandomForestClassifier = pipeline.named_steps["classifier"]
    importances = rf_clf.feature_importances_.tolist()

    # Serialization
    model_path = os.path.join(out_dir, "random_forest.pkl")
    bundle = {
        "model": pipeline,
        "feature_cols": FEATURE_COLUMNS,
        "heuristic": {
            "junk_threshold": junk_threshold,
            "entropy_threshold": entropy_threshold,
        },
        "metadata": {
            "split_method": split_method,
            "test_size": test_size,
            "random_state": random_state,
            "num_features": len(FEATURE_COLUMNS),
        },
    }
    joblib.dump(bundle, model_path)

    metrics = {
        "split_method": split_method,
        "accuracy": ens_acc,
        "precision": ens_prec,
        "recall": ens_rec,
        "f1_score": ens_f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "ml_only": {
            "accuracy": ml_acc,
            "precision": ml_prec,
            "recall": ml_rec,
            "f1_score": ml_f1,
            "roc_auc": ml_auc,
        },
        "heuristic_only": {
            "accuracy": h_acc,
            "precision": h_prec,
            "recall": h_rec,
            "f1_score": h_f1,
        },
        "ensemble": {
            "accuracy": ens_acc,
            "precision": ens_prec,
            "recall": ens_rec,
            "f1_score": ens_f1,
        },
        "cross_validation": {
            "mean_f1": cv_mean,
            "std_f1": cv_std,
        },
        "feature_cols": FEATURE_COLUMNS,
        "feature_importances": importances,
        "samples": {
            "train": int(len(X_train)),
            "test": int(len(X_test)),
            "train_families": int(groups.iloc[train_idx].nunique()),
            "test_families": int(groups.iloc[test_idx].nunique()),
            "zero_leakage_verified": (split_method == "group"),
        },
        "data_provenance": (
            "Evaluated on synthetic JSON PE-like feature records. "
            "Author-reported Microsoft malware dataset and extracted sample variants are external research data "
            "requiring independent provenance documentation before inclusion."
        ),
    }

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved trained model bundle to: %s", model_path)
    logger.info("Saved evaluation metrics to: %s", metrics_path)
    logger.info("Test Ensemble -> Acc: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f", ens_acc, ens_prec, ens_rec, ens_f1)

    return {"model_path": model_path, "metrics_path": metrics_path, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest classifier with group-aware splitting.")
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument("--out", default="detection/model", help="Output dir for model and metrics")
    parser.add_argument(
        "--split",
        choices=["group", "naive"],
        default="group",
        help="Train/test split strategy ('group' prevents lineage leakage; 'naive' for comparison)",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Hold-out test partition size")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    parser.add_argument("--junk-threshold", type=int, default=DEFAULT_HEURISTIC["junk_threshold"], help="Heuristic junk threshold")
    parser.add_argument("--entropy-threshold", type=float, default=DEFAULT_HEURISTIC["entropy_threshold"], help="Heuristic entropy threshold")

    args = parser.parse_args()
    train_model(
        csv_path=args.data,
        out_dir=args.out,
        split_method=args.split,
        test_size=args.test_size,
        random_state=args.seed,
        junk_threshold=args.junk_threshold,
        entropy_threshold=args.entropy_threshold,
    )


if __name__ == "__main__":
    main()
