import os
import json
import uuid
import random
from typing import Dict, List, Optional, Tuple
import click

# Safety: only reads/writes JSON feature records. No binaries touched.
# Simulates polymorphic mutations across generations of feature records.


def _gauss_noise(value: float, sigma: float, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    x = value + random.gauss(0, sigma)
    if min_val is not None:
        x = max(min_val, x)
    if max_val is not None:
        x = min(max_val, x)
    return x


def _mutate_sections(sizes: List[int], entropy: List[float]) -> Tuple[List[int], List[float], List[str]]:
    """Synchronously mutate section sizes and entropy to maintain 1:1 structural alignment."""
    sizes = sizes[:]
    entropy = entropy[:]
    mutations: List[str] = []

    # Pad or trim to equal length if input had historical desynchronization
    target_len = max(len(sizes), len(entropy), 1)
    if len(sizes) < target_len:
        sizes.extend([65536] * (target_len - len(sizes)))
    if len(entropy) < target_len:
        entropy.extend([4.0] * (target_len - len(entropy)))

    if len(sizes) > 0 and random.random() < 0.4 and len(sizes) < 16:
        # Split a random section into two pieces and split entropy accordingly
        idx = random.randrange(len(sizes))
        s = sizes[idx]
        e = entropy[idx]
        if s > 4096:
            a = int(s * random.uniform(0.3, 0.7))
            b = max(1024, s - a)
            sizes[idx] = a
            sizes.insert(idx + 1, b)

            e1 = round(_gauss_noise(e, sigma=0.15, min_val=0.0, max_val=8.0), 3)
            e2 = round(_gauss_noise(e, sigma=0.15, min_val=0.0, max_val=8.0), 3)
            entropy[idx] = e1
            entropy.insert(idx + 1, e2)
            mutations.append("section_split")
    else:
        # Reorder sections together
        combined = list(zip(sizes, entropy))
        random.shuffle(combined)
        sizes = [item[0] for item in combined]
        entropy = [item[1] for item in combined]
        mutations.append("section_reorder")

    # Perturb section entropies with bounded noise
    entropy = [
        round(_gauss_noise(e, sigma=0.2, min_val=0.0, max_val=8.0), 3)
        for e in entropy
    ]
    mutations.append("entropy_noise")

    return sizes, entropy, mutations


def perturb_artifact(art: Dict) -> Dict:
    """Create a polymorphic variant of an artifact with explicit lineage and structural consistency."""
    parent_id = art.get("id")
    root_origin = art.get("origin_id") or parent_id
    parent_gen = int(art.get("generation", 0) or 0)
    label = art.get("label", "benign")

    new_id = str(uuid.uuid4())
    all_mutations: List[str] = []

    # Section mutations
    sizes = art.get("section_sizes") or []
    ent = art.get("section_entropy") or []
    new_sizes, new_ent, sec_mutations = _mutate_sections(sizes, ent)
    all_mutations.extend(sec_mutations)

    # Junk byte padding perturbation (polymorphism injection)
    junk_base = float(art.get("junk_bytes_count", 0) or 0)
    if label == "malware_sim":
        # Malware variants actively manipulate junk code to alter signature
        junk_delta = random.choice([-500, -200, 200, 600, 1200, 2000])
        new_junk = int(max(1500, min(10000, junk_base + junk_delta)))
        all_mutations.append("junk_mutation")
    else:
        # Benign variants experience minor padding changes from compiler/linker options
        new_junk = int(max(0, min(1200, _gauss_noise(junk_base, sigma=80))))

    # Imported APIs: small perturbation (stubs, thunks, or reordered imports)
    api_base = float(art.get("imported_api_count", 50) or 50)
    if label == "malware_sim":
        new_apis = int(max(4, min(55, _gauss_noise(api_base, sigma=4))))
    else:
        new_apis = int(max(20, min(180, _gauss_noise(api_base, sigma=6))))
    all_mutations.append("api_perturbation")

    # Strings count
    str_base = float(art.get("strings_count", 500) or 500)
    if label == "malware_sim":
        new_strings = int(max(15, min(450, _gauss_noise(str_base, sigma=30))))
    else:
        new_strings = int(max(200, min(2500, _gauss_noise(str_base, sigma=70))))

    # Resource size
    res_base = float(art.get("resource_size", 0) or 0)
    new_resource = int(max(0, _gauss_noise(res_base, sigma=6 * 1024)))

    return {
        "id": new_id,
        "origin_id": root_origin,
        "parent_id": parent_id,
        "generation": parent_gen + 1,
        "mutations": all_mutations,
        "label": label,  # Variants strictly inherit their lineage label
        "num_sections": len(new_sizes),
        "section_sizes": new_sizes,
        "section_entropy": new_ent,
        "imported_api_count": new_apis,
        "strings_count": new_strings,
        "entry_point": art.get("entry_point"),
        "image_base": art.get("image_base"),
        "junk_bytes_count": new_junk,
        "resource_size": new_resource,
    }


def synthesize_variants(
    in_dir: str,
    out_dir: str,
    variants_per_file: int,
    seed: Optional[int] = None,
) -> List[str]:
    """Generate polymorphic variants for each base artifact in in_dir.

    Args:
        in_dir: Directory containing base JSON artifacts.
        out_dir: Directory where variant JSONs will be saved.
        variants_per_file: Number of variant descendants to generate per base artifact.
        seed: Optional integer seed for reproducibility.

    Returns:
        List of generated variant JSON file paths.
    """
    if seed is not None:
        random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    created: List[str] = []

    files = [f for f in os.listdir(in_dir) if f.endswith(".json")]
    if not files:
        click.secho(f"No JSON artifacts found in {in_dir}", fg="yellow")
        return []

    for fname in files:
        path = os.path.join(in_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                art = json.load(f)
        except Exception as e:
            click.secho(f"Skipping {fname}: {e}", fg="yellow")
            continue

        for _ in range(variants_per_file):
            var = perturb_artifact(art)
            out_path = os.path.join(out_dir, f"{var['id']}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(var, f, indent=2)
            created.append(out_path)

    click.secho(
        f"Created {len(created)} polymorphic variants from {len(files)} base artifacts in {out_dir}",
        fg="green",
    )
    return created


@click.command()
@click.option("--in", "in_dir", required=True, type=str, help="Input directory of base JSON artifacts")
@click.option("--out", "out_dir", default="dataset/variants", show_default=True, type=str, help="Output directory for variant JSONs")
@click.option("--per", "variants_per_file", default=20, show_default=True, type=int, help="Number of variants per input artifact")
@click.option("--seed", "seed", default=None, type=int, help="Random seed for reproducible synthesis")
def cli(in_dir: str, out_dir: str, variants_per_file: int, seed: Optional[int]) -> None:
    """CLI entrypoint for synthesizing polymorphic variants."""
    synthesize_variants(in_dir, out_dir, variants_per_file, seed=seed)


if __name__ == "__main__":
    cli()
