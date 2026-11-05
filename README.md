# AI_Polymorphism_Detection_Sim

An educational and defensive machine-learning research prototype demonstrating how polymorphic PE variants evade naive signature detection and how a hybrid AI and structural heuristic ensemble defends against evasion.

> **Safety & Research Sandbox Notice:**  
> - **No real binaries** are handled, created, compiled, or executed.
> - **No binary rewriting libraries or malicious toolchains** are invoked.
> - **All artifacts** are synthetic JSON feature records and aggregated CSV files simulating Windows PE structural indicators for defensive research.

---

## Architecture & System Overview

```
                      +-----------------------------+
                      |   Synthetic Base Simulator  |
                      | (artifact_simulator/sim.py) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |     Variant Synthesizer     |
                      |   (Lineage & Mutations)     |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |       Dataset Builder       |
                      |   (build_dataset.py -> CSV) |
                      +--------------+--------------+
                                     |
                                     v
         +---------------------------+---------------------------+
         |                                                       |
         v                                                       v
+-----------------------------+                         +-----------------------------+
|   Group-Aware ML Pipeline   |                         |  Structural Rule Heuristic  |
|  (Random Forest + Scaler)   |                         |  (Entropy & Junk Threshold) |
+--------------+--------------+                         +--------------+--------------+
               |                                                       |
               +---------------------------+---------------------------+
                                           |
                                           v
                            +-----------------------------+
                            |     Hybrid Ensemble & UI    |
                            | (Infer CLI & Web Dashboard) |
                            +-----------------------------+
```

---

## Key Engineering & Research Improvements

* **Group-Aware Hold-Out Splitting:** Resolves variant lineage data leakage where variants of the same parent artifact spanned both training and test partitions. Uses `GroupShuffleSplit` and `GroupKFold` by `origin_id` to evaluate true generalization on unseen polymorphic families.
* **Class-Conditional Generative Model:** Replaced independent random label sampling with principled, class-conditional distributions simulating realistic PE indicators (e.g. elevated entropy and dead-code padding in polymorphic samples vs. standard compiled profiles in benign samples).
* **Synchronous Structural Mutation:** Corrected section splitting in the variant synthesizer to mutate section sizes and section entropies synchronously, guaranteeing 1:1 structural alignment.
* **Unified Detection Architecture:** Implemented `detection.detector.PolymorphicDetector` as a single, thread-safe detection engine shared by `infer.py`, `train_model.py`, and `dashboard/app.py`.
* **Tri-Component Evaluation:** Evaluates and reports metrics independently for the ML model, the heuristic rule, and the hybrid ensemble.
* **Web Dashboard Hardening:** Added upload file size constraints (`MAX_CONTENT_LENGTH = 2MB`), HTTP security headers (`CSP`, `X-Content-Type-Options`, `X-Frame-Options`), strict numeric bounds validation on `/api/predict`, and thread-safe Matplotlib rendering.
* **Automated Test Suite:** 22 unit and integration tests covering the complete simulation and detection lifecycle.

---

## Research Data Sources & Provenance

In accordance with scientific and publication integrity guidelines, this project explicitly distinguishes between its public synthetic simulation datasets and external research sources:

1. **Synthetic PE Feature Records (Public Repository):**  
   All artifacts in `artifact_simulator/sample_artifacts/`, `dataset/variants/`, and `dataset/combined_features.csv` are synthetic JSON records and extracted CSV features simulating PE structural characteristics.
2. **Microsoft Malware Dataset (External Research Corpus):**  
   Referenced in the author's broader research. Specific sample acquisition records, hashes, and provenance manifests are maintained in private research custody. External binary samples are not committed to this public repository to adhere to licensing terms and responsible disclosure practices.
3. **Author-Generated Sample Variant Features (Experimental):**  
   Additional feature vectors extracted from sample variants generated by the author during research. These records are cataloged in private research manifests and are distinct from the public synthetic simulation records.

> *Safe Citation Statement:* The public GitHub pipeline models synthetic PE feature records to evaluate group-aware polymorphic evasion defense. External malware datasets and extracted sample variant datasets are documented separately with appropriate provenance and permission manifests.

---

## Quickstart

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Automated Tests

Run the test suite to verify all modules:
```bash
python -m pytest -v
```

### 3. Generate Synthetic Artifacts & Polymorphic Variants

Generate base artifacts (with reproducible seed):
```bash
python artifact_simulator/simulator.py --count 20 --out artifact_simulator/sample_artifacts --seed 42
```

Synthesize polymorphic variants across generations:
```bash
python variant_synthesizer/synthesize_variants.py --in artifact_simulator/sample_artifacts --out dataset/variants --per 30 --seed 42
```

Build the combined CSV dataset:
```bash
python dataset/build_dataset.py --in artifact_simulator/sample_artifacts --variants dataset/variants --out dataset/combined_features.csv
```

### 4. Train Model with Group-Aware Splitting

Train the leakage-free model:
```bash
python detection/train_model.py --data dataset/combined_features.csv --out detection/model --split group --seed 42
```

*(Optional)* To reproduce and quantify the data leakage inflation caused by naive row-level splitting, run:
```bash
python detection/train_model.py --data dataset/combined_features.csv --out detection/model --split naive --seed 42
```

### 5. Classify a Single Artifact (CLI)

```bash
python detection/infer.py artifact_simulator/sample_artifacts/<some_artifact_id>.json
```

Example JSON output:
```json
{
  "id": "012f0a0b-d62a-45b8-b55c-ced1db6a310a",
  "score": 0.88412,
  "pred": "malware_sim",
  "pred_final": 1,
  "pred_ml": 1,
  "heuristic_flag": 1,
  "explanation": {
    "pred_ml": 1,
    "ml_confidence": 0.8841,
    "heuristic_flag": 1,
    "heuristic_triggered": true,
    "heuristic_thresholds": {
      "junk_threshold": 2500,
      "entropy_threshold": 6.0
    },
    "features": {
      "num_sections": 6,
      "avg_section_entropy": 7.12,
      "total_section_size": 420000,
      "imported_api_count": 18,
      "strings_count": 140,
      "junk_bytes_count": 4500,
      "resource_size": 48000
    }
  }
}
```

### 6. Launch the Security-Hardened Web Dashboard

```bash
python dashboard/app.py
```

Open your browser at `http://127.0.0.1:5000/`.

---

## Dashboard Pages & Features

* **Home (`/`):** Project overview, ethics guidelines, and quick navigation.
* **Metrics (`/metrics`):** Model KPIs, independent ML vs. Heuristic vs. Ensemble comparison table, feature importance chart, confusion matrix, and raw `metrics.json`.
* **Classify (`/classify`):** Secure upload interface for evaluating synthetic `.json` artifacts with structured verdicts.
* **Playground (`/playground`):** Real-time interactive sliders adjusting PE features (`avg_section_entropy`, `junk_bytes_count`, `imported_api_count`, etc.) with AJAX calls to `/api/predict`.
* **Dataset (`/dataset`):** Summary statistics (total samples, lineage families, class balance), dataset schema, and preview of top rows.

---

## REST API Specification

### `POST /api/predict`
Evaluates a numeric feature dictionary and returns classification results.

* **Headers:** `Content-Type: application/json`
* **Request Body:**
  ```json
  {
    "num_sections": 5,
    "avg_section_entropy": 6.8,
    "total_section_size": 250000,
    "imported_api_count": 20,
    "strings_count": 180,
    "junk_bytes_count": 3500,
    "resource_size": 45000
  }
  ```
* **Success Response (200 OK):**
  ```json
  {
    "score": 0.912,
    "pred": "malware_sim",
    "pred_final": 1,
    "pred_ml": 1,
    "heuristic_flag": 1,
    "explanation": { ... }
  }
  ```
* **Validation Error (400 Bad Request):**
  ```json
  {
    "error": "Value for avg_section_entropy out of allowed bounds [0.0, 8.0]: 12.5"
  }
  ```

---

## Project Structure

```
AI_Polymorphism_Detection_Sim/
├── artifact_simulator/
│   ├── __init__.py
│   ├── sample_artifacts/        # Generated base JSON artifacts
│   └── simulator.py             # Class-conditional synthetic generator
├── dashboard/
│   ├── __init__.py
│   └── app.py                   # Hardened Flask web application
├── dataset/
│   ├── __init__.py
│   ├── build_dataset.py         # JSON-to-CSV dataset compiler
│   ├── combined_features.csv    # Flattened feature records
│   └── variants/                # Generated polymorphic variant JSONs
├── detection/
│   ├── __init__.py
│   ├── detector.py              # Unified PolymorphicDetector engine
│   ├── infer.py                 # Single-artifact CLI inference
│   ├── train_model.py           # Group-aware training pipeline
│   └── model/
│       ├── metrics.json         # Evaluation metrics and metadata
│       └── random_forest.pkl    # Serialized pipeline and heuristic bundle
├── docs/
│   └── TECHNICAL_AUDIT.md       # Comprehensive engineering & security audit
├── notebooks/
│   └── analysis.ipynb           # Exploratory data analysis notebook
├── tests/
│   ├── conftest.py              # Pytest fixtures and mock pipeline
│   ├── test_build_dataset.py
│   ├── test_dashboard.py
│   ├── test_detector.py
│   ├── test_infer.py
│   ├── test_simulator.py
│   └── test_synthesizer.py
├── ethics.txt                   # Ethical guidelines and sandbox disclaimer
├── requirements.txt             # Project dependencies
└── README.md
```

---

## Technical Audit Report

For in-depth mathematical analysis, root-cause investigations, benchmark comparisons between naive and group-aware splitting, and remaining limitations, see [`docs/TECHNICAL_AUDIT.md`](docs/TECHNICAL_AUDIT.md).

---

## Ethics and Responsible Research

* This software is intended strictly for educational, defensive, and academic research purposes.
* It does not contain exploit payloads, shellcode, or malicious capabilities.
* Testing against real-world binaries must only be conducted in an isolated, air-gapped sandbox with appropriate administrative authorization. See [`ethics.txt`](ethics.txt).
