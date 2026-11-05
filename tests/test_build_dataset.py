import os
import json
import pandas as pd
import pytest
from dataset.build_dataset import build_dataset
from artifact_simulator.simulator import generate_single_artifact


def test_build_dataset(tmp_path):
    """Verify build_dataset creates a valid CSV with correct schema and no null origin_ids."""
    art_dir = str(tmp_path / "artifacts")
    var_dir = str(tmp_path / "variants")
    out_csv = str(tmp_path / "output.csv")
    os.makedirs(art_dir, exist_ok=True)
    os.makedirs(var_dir, exist_ok=True)

    base = generate_single_artifact(target_label="benign")
    with open(os.path.join(art_dir, f"{base['id']}.json"), "w", encoding="utf-8") as f:
        json.dump(base, f)

    var = {
        "id": "variant-123",
        "origin_id": base["id"],
        "parent_id": base["id"],
        "generation": 1,
        "label": "benign",
        "num_sections": 4,
        "section_sizes": [1024, 2048, 4096, 8192],
        "section_entropy": [3.1, 4.2, 5.0, 2.8],
        "imported_api_count": 55,
        "strings_count": 600,
        "junk_bytes_count": 100,
        "resource_size": 20480,
    }
    with open(os.path.join(var_dir, "variant-123.json"), "w", encoding="utf-8") as f:
        json.dump(var, f)

    csv_path = build_dataset(artifacts_dir=art_dir, variants_dir=var_dir, out_csv=out_csv)
    assert os.path.isfile(csv_path)

    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert "origin_id" in df.columns
    assert df["origin_id"].isna().sum() == 0
    assert df["origin_id"].nunique() == 1  # Both belong to the same family!
    assert "avg_section_entropy" in df.columns
    assert df["avg_section_entropy"].isna().sum() == 0

