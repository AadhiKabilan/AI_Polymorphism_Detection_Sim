import os
import sys
import json
import argparse
from typing import Dict, Any

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from detection.detector import PolymorphicDetector

# Safety: Works only with JSON feature records and saved sklearn model bundle.


def classify_artifact(json_file: str, model_path: str = "detection/model/random_forest.pkl") -> Dict[str, Any]:
    """Classify a single PE-like JSON artifact using the saved detection bundle."""
    if not os.path.isfile(json_file):
        raise FileNotFoundError(f"Artifact JSON not found at: {json_file}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Train the model first.")

    with open(json_file, "r", encoding="utf-8") as f:
        rec = json.load(f)

    detector = PolymorphicDetector.load(model_path)
    result = detector.predict_record(rec)

    # Output formatted JSON to stdout
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description="Classify a single JSON feature artifact.")
    parser.add_argument("artifact", help="Path to JSON artifact to classify")
    parser.add_argument(
        "--model",
        default=os.path.join("detection", "model", "random_forest.pkl"),
        help="Path to saved model bundle (default: detection/model/random_forest.pkl)",
    )
    args = parser.parse_args()

    try:
        classify_artifact(args.artifact, args.model)
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
