import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

FEATURE_COLUMNS: List[str] = [
    "num_sections",
    "avg_section_entropy",
    "total_section_size",
    "imported_api_count",
    "strings_count",
    "junk_bytes_count",
    "resource_size",
]

DEFAULT_HEURISTIC = {
    "junk_threshold": 2500,
    "entropy_threshold": 6.0,
}


def extract_features_from_dict(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Safely extract and aggregate standardized PE-like features from a JSON record.

    Handles missing keys, empty section lists, type coercion, and lineage IDs.
    """
    if not isinstance(rec, dict):
        raise ValueError(f"Expected dict record, got {type(rec).__name__}")

    sizes = rec.get("section_sizes") or []
    ent = rec.get("section_entropy") or []

    # Clean numeric lists
    try:
        clean_sizes = [int(s) for s in sizes if s is not None and not np.isnan(float(s))]
    except (ValueError, TypeError):
        clean_sizes = []

    try:
        clean_ent = [float(e) for e in ent if e is not None and not np.isnan(float(e))]
    except (ValueError, TypeError):
        clean_ent = []

    # Determine num_sections
    raw_num = rec.get("num_sections")
    if raw_num is not None:
        try:
            num_sections = max(0, int(raw_num))
        except (ValueError, TypeError):
            num_sections = max(len(clean_sizes), len(clean_ent))
    else:
        num_sections = max(len(clean_sizes), len(clean_ent))

    avg_entropy = float(np.mean(clean_ent)) if len(clean_ent) > 0 else 0.0
    total_size = int(np.sum(clean_sizes)) if len(clean_sizes) > 0 else 0

    def _safe_int(val: Any, default: int = 0) -> int:
        try:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return max(0, int(val))
        except (ValueError, TypeError):
            return default

    record_id = rec.get("id")
    origin_id = rec.get("origin_id")
    # If origin_id is absent or empty, the record is its own lineage root
    effective_origin = origin_id if (origin_id is not None and str(origin_id).strip() != "") else record_id

    return {
        "id": record_id,
        "origin_id": effective_origin,
        "parent_id": rec.get("parent_id"),
        "generation": _safe_int(rec.get("generation"), 0),
        "num_sections": num_sections,
        "avg_section_entropy": round(avg_entropy, 6),
        "total_section_size": total_size,
        "imported_api_count": _safe_int(rec.get("imported_api_count"), 0),
        "strings_count": _safe_int(rec.get("strings_count"), 0),
        "junk_bytes_count": _safe_int(rec.get("junk_bytes_count"), 0),
        "resource_size": _safe_int(rec.get("resource_size"), 0),
        "label": rec.get("label", "benign"),
    }


def evaluate_heuristic(
    features: Dict[str, Any],
    junk_threshold: int = DEFAULT_HEURISTIC["junk_threshold"],
    entropy_threshold: float = DEFAULT_HEURISTIC["entropy_threshold"],
) -> int:
    """Evaluate rule-based heuristic for simulated polymorphic evasion indicators.

    Fires (returns 1) if junk padding and section entropy both exceed their respective thresholds.
    """
    junk = float(features.get("junk_bytes_count", 0) or 0)
    entropy = float(features.get("avg_section_entropy", 0.0) or 0.0)
    return int((junk > junk_threshold) and (entropy > entropy_threshold))


class PolymorphicDetector:
    """Thread-safe detector encapsulating ML pipeline and heuristic verification."""

    def __init__(self, bundle: Dict[str, Any]):
        if not isinstance(bundle, dict):
            raise ValueError("Model bundle must be a dictionary.")
        if "model" not in bundle:
            raise KeyError("Model bundle missing required 'model' key.")

        self.model = bundle["model"]
        self.feature_cols: List[str] = bundle.get("feature_cols", FEATURE_COLUMNS)
        self.heuristic_config: Dict[str, Any] = bundle.get("heuristic", DEFAULT_HEURISTIC)
        self.metadata: Dict[str, Any] = bundle.get("metadata", {})

    @classmethod
    def load(cls, model_path: str) -> "PolymorphicDetector":
        """Load detector bundle from a serialized .pkl file."""
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        bundle = joblib.load(model_path)
        return cls(bundle)

    def predict_features(self, flat_features: Dict[str, Any]) -> Dict[str, Any]:
        """Perform combined inference from extracted numeric feature dictionary."""
        # Validate and construct input vector in expected order
        vec: List[float] = []
        for col in self.feature_cols:
            val = flat_features.get(col, 0.0)
            try:
                fval = float(val) if val is not None else 0.0
                if np.isnan(fval) or np.isinf(fval):
                    fval = 0.0
            except (ValueError, TypeError):
                fval = 0.0
            vec.append(fval)

        X = np.array([vec], dtype=float)

        # ML Probability
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            # Binary classification assumed: class 1 is malware_sim
            if proba.shape[1] >= 2:
                score = float(proba[0, 1])
            else:
                score = float(proba[0, 0])
        elif hasattr(self.model, "decision_function"):
            decision = float(self.model.decision_function(X)[0])
            score = 1.0 / (1.0 + np.exp(-decision))
        else:
            score = float(self.model.predict(X)[0])

        pred_ml = int(score >= 0.5)

        # Heuristic evaluation
        junk_th = self.heuristic_config.get("junk_threshold", DEFAULT_HEURISTIC["junk_threshold"])
        ent_th = self.heuristic_config.get("entropy_threshold", DEFAULT_HEURISTIC["entropy_threshold"])
        h_flag = evaluate_heuristic(flat_features, junk_threshold=junk_th, entropy_threshold=ent_th)

        pred_final = int(max(pred_ml, h_flag))
        pred_label = "malware_sim" if pred_final == 1 else "benign"

        explanation = {
            "pred_ml": pred_ml,
            "ml_confidence": round(score, 4),
            "heuristic_flag": h_flag,
            "heuristic_triggered": bool(h_flag),
            "heuristic_thresholds": {"junk_threshold": junk_th, "entropy_threshold": ent_th},
            "features": {col: flat_features.get(col) for col in self.feature_cols},
        }

        return {
            "id": flat_features.get("id"),
            "score": round(score, 6),
            "pred": pred_label,
            "pred_final": pred_final,
            "pred_ml": pred_ml,
            "heuristic_flag": h_flag,
            "explanation": explanation,
        }

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from arbitrary raw JSON record and predict."""
        flat = extract_features_from_dict(record)
        return self.predict_features(flat)

