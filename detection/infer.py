import os
import json
import argparse
import joblib
import numpy as np
import pandas as pd

# Safety: Works only with JSON features and saved sklearn model.


def _flatten_features(rec):
    sizes = rec.get("section_sizes", []) or []
    ent = rec.get("section_entropy", []) or []
    num_sections = int(rec.get("num_sections", len(sizes) or len(ent)))
    avg_entropy = float(np.mean(ent)) if len(ent) > 0 else 0.0
    total_size = int(np.sum(sizes)) if len(sizes) > 0 else 0
    return {
        "id": rec.get("id"),
        "origin_id": rec.get("origin_id"),
        "num_sections": num_sections,
        "avg_section_entropy": avg_entropy,
        "total_section_size": total_size,
        "imported_api_count": int(rec.get("imported_api_count", 0)),
        "strings_count": int(rec.get("strings_count", 0)),
        "junk_bytes_count": int(rec.get("junk_bytes_count", 0)),
        "resource_size": int(rec.get("resource_size", 0)),
        "label": rec.get("label"),
    }


def classify_artifact(json_file: str, model_path: str) -> dict:
    with open(json_file, "r", encoding="utf-8") as f:
        rec = json.load(f)
    flat = _flatten_features(rec)

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    heuristic = bundle.get("heuristic", {"junk_threshold": 2500, "entropy_threshold": 6.0})

    df = pd.DataFrame([flat])
    X = df[feature_cols].values.astype(float)
    score = float(model.predict_proba(X)[:, 1][0])
    pred_ml = int(score >= 0.5)

    h_flag = int((flat["junk_bytes_count"] > heuristic.get("junk_threshold", 2500)) and (flat["avg_section_entropy"] > heuristic.get("entropy_threshold", 6.0)))
    pred_final = int(max(pred_ml, h_flag))

    result = {
        "id": flat.get("id"),
        "score": round(score, 6),
        "pred": "malware_sim" if pred_final == 1 else "benign",
        "explanation": {
            "pred_ml": pred_ml,
            "heuristic_flag": h_flag,
            "junk_bytes_count": flat["junk_bytes_count"],
            "avg_section_entropy": round(flat["avg_section_entropy"], 6),
            "imported_api_count": flat["imported_api_count"],
            "strings_count": flat["strings_count"],
            "total_section_size": flat["total_section_size"],
        },
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", help="Path to JSON artifact to classify")
    parser.add_argument("--model", default=os.path.join("detection", "model", "random_forest.pkl"), help="Path to saved model bundle")
    args = parser.parse_args()
    classify_artifact(args.artifact, args.model)


if __name__ == "__main__":
    # Example minimal usage:
    #   python detection/infer.py artifact_simulator/sample_artifacts/<some_id>.json
    try:
        main()
    except SystemExit:
        pass
