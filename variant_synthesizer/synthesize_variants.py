import os
import json
import uuid
import random
import math
from typing import Dict, List
import click

# Safety: only reads/writes JSON feature records. No binaries touched.


def _gauss_noise(value: float, sigma: float, min_val: float = None, max_val: float = None) -> float:
    x = value + random.gauss(0, sigma)
    if min_val is not None:
        x = max(min_val, x)
    if max_val is not None:
        x = min(max_val, x)
    return x


def _split_or_shuffle_sizes(sizes: List[int]) -> List[int]:
    sizes = sizes[:]
    if len(sizes) > 0 and random.random() < 0.5:
        # Split a random section into two pieces
        idx = random.randrange(len(sizes))
        s = sizes[idx]
        if s > 2048:
            a = int(s * random.uniform(0.3, 0.7))
            b = max(1, s - a)
            sizes[idx] = a
            sizes.insert(idx + 1, b)
    else:
        random.shuffle(sizes)
    return sizes


def _perturb_artifact(art: Dict) -> Dict:
    new_art = dict(art)
    new_art["id"] = str(uuid.uuid4())
    new_art["origin_id"] = art.get("id")

    # Entropy perturbation
    ent = art.get("section_entropy", [])
    new_art["section_entropy"] = [
        round(_gauss_noise(e, sigma=0.25, min_val=0.0, max_val=8.0), 3) for e in ent
    ]

    # Size perturbation
    sizes = art.get("section_sizes", [])
    new_art["section_sizes"] = _split_or_shuffle_sizes(sizes)
    new_art["num_sections"] = len(new_art["section_sizes"]) if new_art.get("section_sizes") else art.get("num_sections", 0)

    # Junk bytes: slight increase or decrease
    junk = int(max(0, _gauss_noise(float(art.get("junk_bytes_count", 0)), sigma=500)))
    # Occasionally add extra junk to mimic polymorphism padding
    if random.random() < 0.3:
        junk += random.randint(100, 3000)
    new_art["junk_bytes_count"] = junk

    # Imported APIs: small random walk
    apis = art.get("imported_api_count", 0)
    apis = int(max(0, _gauss_noise(float(apis), sigma=5)))
    if random.random() < 0.2:
        # Optional reordering proxy: small spike or drop
        apis += random.choice([-10, -5, 5, 10])
        apis = max(0, apis)
    new_art["imported_api_count"] = apis

    # Strings count: add/remove benign-leaning strings
    strings = art.get("strings_count", 0)
    delta = int(random.gauss(0, 50))
    new_art["strings_count"] = max(0, strings + delta)

    # Do not change label by default
    new_art["label"] = art.get("label", "benign")

    # Keep other fields consistent
    new_art["entry_point"] = art.get("entry_point")
    new_art["image_base"] = art.get("image_base")
    new_art["resource_size"] = int(max(0, _gauss_noise(float(art.get("resource_size", 0)), sigma=8 * 1024)))

    return new_art


def synthesize_variants(in_dir: str, out_dir: str, variants_per_file: int) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    created: List[str] = []
    files = [f for f in os.listdir(in_dir) if f.endswith('.json')]
    for fname in files:
        path = os.path.join(in_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                art = json.load(f)
        except Exception as e:
            click.secho(f"Skipping {fname}: {e}", fg="yellow")
            continue
        for _ in range(variants_per_file):
            var = _perturb_artifact(art)
            out_path = os.path.join(out_dir, f"{var['id']}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(var, f, indent=2)
            created.append(out_path)
    click.secho(f"Created {len(created)} variant files in {out_dir}", fg="green")
    return created


@click.command()
@click.option("--in", "in_dir", required=True, type=str, help="Input directory of base JSON artifacts")
@click.option("--out", "out_dir", default="dataset/variants", show_default=True, type=str, help="Output directory for variant JSONs")
@click.option("--per", "variants_per_file", default=20, show_default=True, type=int, help="Number of variants per input artifact")
def cli(in_dir: str, out_dir: str, variants_per_file: int) -> None:
    synthesize_variants(in_dir, out_dir, variants_per_file)


if __name__ == "__main__":
    # Example minimal run:
    #   python variant_synthesizer/synthesize_variants.py --in artifact_simulator/sample_artifacts --out dataset/variants --per 5
    cli(standalone_mode=False)
    try:
        default_in = os.path.join("artifact_simulator", "sample_artifacts")
        default_out = os.path.join("dataset", "variants")
        if os.path.isdir(default_in):
            synthesize_variants(default_in, default_out, 3)
    except Exception as e:
        click.secho(f"Demo synthesis encountered an issue: {e}", fg="yellow")
