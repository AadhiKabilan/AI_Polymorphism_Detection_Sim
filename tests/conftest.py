import os
import sys
import tempfile
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from artifact_simulator.simulator import generate_base_artifacts
from variant_synthesizer.synthesize_variants import synthesize_variants
from dataset.build_dataset import build_dataset
from detection.train_model import train_model
from dashboard.app import app


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Generate a clean, fast synthetic dataset and trained model for testing."""
    base_tmp = tmp_path_factory.mktemp("poly_sim_data")
    art_dir = str(base_tmp / "artifacts")
    var_dir = str(base_tmp / "variants")
    csv_path = str(base_tmp / "combined.csv")
    model_dir = str(base_tmp / "model")

    # Generate 20 base artifacts (10 benign, 10 malware_sim) with fixed seed
    generate_base_artifacts(n=20, out_dir=art_dir, seed=42, benign_ratio=0.5)

    # Generate 3 variants per base artifact (60 variants)
    synthesize_variants(in_dir=art_dir, out_dir=var_dir, variants_per_file=3, seed=42)

    # Build CSV dataset (80 total rows across 20 families)
    build_dataset(artifacts_dir=art_dir, variants_dir=var_dir, out_csv=csv_path)

    # Train model with group-aware split
    train_model(csv_path=csv_path, out_dir=model_dir, split_method="group", random_state=42)

    return {
        "art_dir": art_dir,
        "var_dir": var_dir,
        "csv_path": csv_path,
        "model_dir": model_dir,
        "model_file": os.path.join(model_dir, "random_forest.pkl"),
        "metrics_file": os.path.join(model_dir, "metrics.json"),
    }


@pytest.fixture
def flask_client(test_data_dir):
    """Provide Flask test client pointing to test artifacts."""
    app.config["MODEL_DIR"] = test_data_dir["model_dir"]
    app.config["MODEL_PATH"] = test_data_dir["model_file"]
    app.config["METRICS_PATH"] = test_data_dir["metrics_file"]
    app.config["DATASET_CSV"] = test_data_dir["csv_path"]
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client
