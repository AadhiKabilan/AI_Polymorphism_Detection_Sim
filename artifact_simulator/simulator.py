import os
import json
import uuid
import random
from typing import List, Dict
import click

# Safety note: This module only creates synthetic JSON feature records.
# It does not create, modify, compile, or execute any binaries.


def _generate_single_artifact() -> Dict:
    num_sections = random.randint(3, 10)
    # Section sizes in KB, convert to bytes to be more realistic
    section_sizes_kb = [random.randint(8, 1024) for _ in range(num_sections)]
    section_sizes = [size_kb * 1024 for size_kb in section_sizes_kb]
    section_entropy = [round(random.uniform(0.2, 8.0), 3) for _ in range(num_sections)]

    imported_api_count = max(0, int(random.gauss(50, 15)))
    strings_count = max(0, int(random.gauss(500, 200)))
    entry_point = random.randint(0x1000, 0x400000)
    image_base = random.choice([0x400000, 0x10000000, 0x140000000])
    junk_bytes_count = max(0, int(abs(random.gauss(2000, 1500))))
    resource_size = max(0, int(abs(random.gauss(50 * 1024, 40 * 1024))))

    # Assign label with a skew to have both classes
    label = random.choices(["benign", "malware_sim"], weights=[0.6, 0.4], k=1)[0]

    return {
        "id": str(uuid.uuid4()),
        "label": label,
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


def generate_base_artifacts(n: int, out_dir: str) -> List[str]:
    """Generate n synthetic artifacts and save as JSON files.

    Returns list of file paths created.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths: List[str] = []
    for _ in range(n):
        art = _generate_single_artifact()
        path = os.path.join(out_dir, f"{art['id']}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(art, f, indent=2)
        paths.append(path)
    click.secho(f"Generated {len(paths)} artifacts in {out_dir}", fg="green")
    return paths


@click.command()
@click.option("--count", "count", default=50, show_default=True, type=int, help="Number of artifacts to generate")
@click.option("--out", "out_dir", default="artifact_simulator/sample_artifacts", show_default=True, type=str, help="Output directory for JSON artifacts")
def cli(count: int, out_dir: str) -> None:
    """CLI entrypoint for generating base synthetic artifacts."""
    generate_base_artifacts(count, out_dir)


if __name__ == "__main__":
    # Example: small demonstration run if executed directly without CLI args
    # You can also run the CLI:
    #   python artifact_simulator/simulator.py --count 20 --out artifact_simulator/sample_artifacts
    cli(standalone_mode=False)
    # Fallback demo when run with no args in some environments
    try:
        generate_base_artifacts(5, os.path.join("artifact_simulator", "sample_artifacts"))
    except Exception as e:
        click.secho(f"Demo generation encountered an issue: {e}", fg="yellow")
