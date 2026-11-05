import os
import json
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import click

import sys
# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from detection.detector import extract_features_from_dict, FEATURE_COLUMNS

# Safety: Reads JSON feature files and writes CSV. No binaries handled.


def _collect_jsons_from_dirs(dirs: List[str]) -> List[Dict]:
    """Read all JSON artifacts across provided directories and extract flat feature records."""
    rows: List[Dict] = []
    seen_ids = set()

    for p in dirs:
        if not os.path.isdir(p):
            continue
        for fname in os.listdir(p):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(p, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    rec = json.load(f)
                flat = extract_features_from_dict(rec)
                art_id = flat.get("id")
                if art_id and art_id in seen_ids:
                    continue
                if art_id:
                    seen_ids.add(art_id)
                rows.append(flat)
            except Exception as e:
                click.secho(f"Skipping {fpath}: {e}", fg="yellow")

    return rows


def build_dataset(artifacts_dir: str, variants_dir: str, out_csv: str) -> str:
    """Collect base artifacts and variants, extract features, and output a validated CSV.

    Ensures consistent columns, valid non-null lineage IDs, and data validation.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    rows = _collect_jsons_from_dirs([artifacts_dir, variants_dir])

    if not rows:
        raise ValueError(f"No valid JSON records found in {artifacts_dir} or {variants_dir}")

    df = pd.DataFrame(rows)

    # Standard columns with full lineage tracking
    columns = [
        "id",
        "origin_id",
        "parent_id",
        "generation",
        "num_sections",
        "avg_section_entropy",
        "total_section_size",
        "imported_api_count",
        "strings_count",
        "junk_bytes_count",
        "resource_size",
        "label",
    ]

    # Ensure all columns exist
    for col in columns:
        if col not in df.columns:
            df[col] = None

    df = df[columns]

    # Ensure origin_id is never empty/null (fallback to record id for base artifacts)
    df["origin_id"] = df["origin_id"].fillna(df["id"])

    # Clean numeric types
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df.to_csv(out_csv, index=False)

    class_dist = df["label"].value_counts().to_dict()
    num_families = df["origin_id"].nunique()

    click.secho(
        f"Wrote dataset with {len(df)} rows across {num_families} lineage families to {out_csv}",
        fg="green",
    )
    click.secho(f"Class distribution: {class_dist}", fg="cyan")

    return out_csv


@click.command()
@click.option("--in", "artifacts_dir", required=True, type=str, help="Directory with base artifacts JSON")
@click.option("--variants", "variants_dir", required=True, type=str, help="Directory with variant JSON files")
@click.option("--out", "out_csv", default="dataset/combined_features.csv", show_default=True, type=str, help="Output CSV path")
def cli(artifacts_dir: str, variants_dir: str, out_csv: str) -> None:
    """CLI entrypoint for building combined dataset CSV."""
    build_dataset(artifacts_dir, variants_dir, out_csv)


if __name__ == "__main__":
    cli()
