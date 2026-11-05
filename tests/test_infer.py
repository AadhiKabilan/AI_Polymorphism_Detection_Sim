import os
import json
import pytest
from detection.infer import classify_artifact


def test_classify_artifact(test_data_dir):
    """Verify single-artifact inference CLI logic."""
    # Find a sample artifact
    files = [f for f in os.listdir(test_data_dir["art_dir"]) if f.endswith(".json")]
    assert len(files) > 0
    sample_file = os.path.join(test_data_dir["art_dir"], files[0])

    res = classify_artifact(sample_file, test_data_dir["model_file"])
    assert "pred" in res
    assert res["pred"] in ("benign", "malware_sim")
    assert "score" in res
    assert 0.0 <= res["score"] <= 1.0
    assert "explanation" in res


def test_classify_artifact_missing_file(test_data_dir):
    """Verify proper exception on missing input artifact."""
    with pytest.raises(FileNotFoundError):
        classify_artifact("non_existent_file.json", test_data_dir["model_file"])

