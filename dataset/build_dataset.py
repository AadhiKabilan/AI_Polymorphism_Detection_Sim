import os
import json
from typing import List, Dict
import pandas as pd
import numpy as np
import click

# Safety: Reads JSON feature files and writes CSV. No binaries handled.


def _flatten_features(rec: Dict) -> Dict:
    sizes = rec.get("section_sizes", []) or []
    ent = rec.get("section_entropy", []) or []
    num_sections = int(rec.get("num_sections", len(sizes)))
    if not num_sections:
        num_sections = len(sizes) if sizes else len(ent)

    avg_entropy = float(np.mean(ent)) if len(ent) > 0 else 0.0
    total_size = int(np.sum(sizes)) if len(sizes) > 0 else 0

    return {
        "id": rec.get("id"),
        "origin_id": rec.get("origin_id"),
        "num_sections": num_sections,
        "avg_section_entropy": round(avg_entropy, 6),
        "total_section_size": total_size,
        "imported_api_count": int(rec.get("imported_api_count", 0)),
        "strings_count": int(rec.get("strings_count", 0)),
        "junk_bytes_count": int(rec.get("junk_bytes_count", 0)),
        "resource_size": int(rec.get("resource_size", 0)),
        "label": rec.get("label", "benign"),
    }


def _collect_jsons(paths: List[str]) -> List[Dict]:
    rows: List[Dict] = []
    for p in paths:
        if not os.path.isdir(p):
            continue
        for fname in os.listdir(p):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(p, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                rows.append(_flatten_features(rec))
            except Exception as e:
                click.secho(f"Skipping {fpath}: {e}", fg="yellow")
    return rows


def build_dataset(artifacts_dir: str, variants_dir: str, out_csv: str) -> str:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows = _collect_jsons([artifacts_dir, variants_dir])
    df = pd.DataFrame(rows)
    # Ensure consistent column order
    columns = [
        "id",
        "origin_id",
        "num_sections",
        "avg_section_entropy",
        "total_section_size",
        "imported_api_count",
        "strings_count",
        "junk_bytes_count",
        "resource_size",
        "label",
    ]
    df = df[columns]
    df.to_csv(out_csv, index=False)
    click.secho(f"Wrote dataset with {len(df)} rows to {out_csv}", fg="green")
    return out_csv


@click.command()
@click.option("--in", "artifacts_dir", required=True, type=str, help="Directory with base artifacts JSON")
@click.option("--variants", "variants_dir", required=True, type=str, help="Directory with variant JSON files")
@click.option("--out", "out_csv", default="dataset/combined_features.csv", show_default=True, type=str, help="Output CSV path")
def cli(artifacts_dir: str, variants_dir: str, out_csv: str) -> None:
    build_dataset(artifacts_dir, variants_dir, out_csv)


if __name__ == "__main__":
    # Example minimal run:
    #   python dataset/build_dataset.py --in artifact_simulator/sample_artifacts --variants dataset/variants --out dataset/combined_features.csv
    cli(standalone_mode=False)
    try:
        default_art = os.path.join("artifact_simulator", "sample_artifacts")
        default_var = os.path.join("dataset", "variants")
        default_out = os.path.join("dataset", "combined_features.csv")
        if os.path.isdir(default_art) and os.path.isdir(default_var):
            build_dataset(default_art, default_var, default_out)
    except Exception as e:
        click.secho(f"Demo dataset build encountered an issue: {e}", fg="yellow")
