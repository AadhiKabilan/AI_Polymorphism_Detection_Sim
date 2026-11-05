import os
import json
import uuid
import random
from typing import List, Dict, Optional
import click

# Safety note: This module only creates synthetic JSON feature records.
# It does not create, modify, compile, or execute any binaries.
# The synthetic records simulate PE structural indicators for educational and defensive evaluation.


def generate_single_artifact(target_label: Optional[str] = None) -> Dict:
    """Generate a single synthetic PE-like feature record with class-conditional distributions.

    Creates a principled, explainable relationship between simulated structural characteristics
    (e.g., entropy, junk padding, imported APIs, and strings) and simulated class labels.
    """
    if target_label is None:
        target_label = random.choices(["benign", "malware_sim"], weights=[0.5, 0.5], k=1)[0]

    art_id = str(uuid.uuid4())

    if target_label == "malware_sim":
        # Simulated polymorphic / packed malware profile:
        # - High section entropy (due to encryption/packing)
        # - Extensive junk byte / dead code injection (evasion padding)
        # - Reduced direct API imports (dynamic resolution / stealth loading)
        # - Suppressed readable strings (obfuscated or packed content)
        num_sections = random.randint(4, 9)
        section_sizes_kb = [random.randint(16, 1200) for _ in range(num_sections)]
        section_sizes = [s * 1024 for s in section_sizes_kb]

        # Clustered in high entropy range (e.g. packed code / encrypted payloads)
        section_entropy = [
            round(min(7.98, max(5.2, random.gauss(6.85, 0.55))), 3)
            for _ in range(num_sections)
        ]

        imported_api_count = max(4, min(50, int(random.gauss(22, 8))))
        strings_count = max(20, min(400, int(random.gauss(160, 65))))
        junk_bytes_count = max(1800, min(9500, int(random.gauss(4200, 1300))))
        resource_size = max(0, int(abs(random.gauss(75 * 1024, 45 * 1024))))

    else:
        # Simulated benign executable profile:
        # - Moderate, realistic section entropy (standard compiled code and data)
        # - Minimal or zero junk byte padding
        # - Normal to high direct API imports from standard system libraries
        # - Normal readable strings (messages, formats, symbols)
        num_sections = random.randint(3, 6)
        section_sizes_kb = [random.randint(16, 1024) for _ in range(num_sections)]
        section_sizes = [s * 1024 for s in section_sizes_kb]

        # Clustered in standard compiled PE range
        section_entropy = [
            round(min(6.2, max(0.5, random.gauss(4.1, 0.8))), 3)
            for _ in range(num_sections)
        ]

        imported_api_count = max(25, min(160, int(random.gauss(70, 18))))
        strings_count = max(250, min(2200, int(random.gauss(750, 220))))
        junk_bytes_count = max(0, min(1000, int(abs(random.gauss(250, 200)))))
        resource_size = max(0, int(abs(random.gauss(45 * 1024, 30 * 1024))))

    entry_point = random.randint(0x1000, 0x400000)
    image_base = random.choice([0x400000, 0x10000000, 0x140000000])

    return {
        "id": art_id,
        "origin_id": art_id,  # Lineage root for base artifact
        "parent_id": None,
        "generation": 0,
        "label": target_label,
        "num_sections": num_sections,
        "section_sizes": section_sizes,
        "section_entropy": section_entropy,
        "imported_api_count": imported_api_count,
        "strings_count": strings_count,
        "entry_point": entry_point,
        "image_base": image_base,
        "junk_bytes_count": junk_bytes_count,
        "resource_size": resource_size,
    }


def generate_base_artifacts(
    n: int,
    out_dir: str,
    seed: Optional[int] = None,
    benign_ratio: float = 0.5,
) -> List[str]:
    """Generate n synthetic base artifacts and save as individual JSON records.

    Args:
        n: Number of artifacts to generate.
        out_dir: Directory where JSON files will be written.
        seed: Optional integer seed for reproducible generation.
        benign_ratio: Proportion of benign artifacts (0.0 to 1.0).

    Returns:
        List of created JSON file paths.
    """
    if seed is not None:
        random.seed(seed)

    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []

    benign_count = int(round(n * benign_ratio))
    labels = ["benign"] * benign_count + ["malware_sim"] * (n - benign_count)
    random.shuffle(labels)

    for i in range(n):
        label = labels[i]
        art = generate_single_artifact(target_label=label)
        path = os.path.join(out_dir, f"{art['id']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(art, f, indent=2)
        paths.append(path)

    click.secho(
        f"Generated {len(paths)} synthetic base artifacts in {out_dir} "
        f"({benign_count} benign, {n - benign_count} malware_sim)",
        fg="green",
    )
    return paths


@click.command()
@click.option("--count", "count", default=50, show_default=True, type=int, help="Number of artifacts to generate")
@click.option("--out", "out_dir", default="artifact_simulator/sample_artifacts", show_default=True, type=str, help="Output directory for JSON artifacts")
@click.option("--seed", "seed", default=None, type=int, help="Random seed for reproducible generation")
@click.option("--benign-ratio", "benign_ratio", default=0.5, show_default=True, type=float, help="Ratio of benign samples (0.0 to 1.0)")
def cli(count: int, out_dir: str, seed: Optional[int], benign_ratio: float) -> None:
    """CLI entrypoint for generating base synthetic artifacts."""
    generate_base_artifacts(count, out_dir, seed=seed, benign_ratio=benign_ratio)


if __name__ == "__main__":
    cli()
