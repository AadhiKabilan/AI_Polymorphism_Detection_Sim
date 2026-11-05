# Technical Audit & Architecture Remediations

**Project:** AI_Polymorphism_Detection_Sim  
**Repository:** [https://github.com/AadhiKabilan/AI_Polymorphism_Detection_Sim](https://github.com/AadhiKabilan/AI_Polymorphism_Detection_Sim)  
**Author:** Aadhi Kabilan J  
**Date:** November 2026  
**Auditor Role:** Senior Python, ML, and Cybersecurity Engineer  

---

## 1. Executive Summary

This document details the comprehensive engineering audit, security review, and architecture overhaul implemented across the `AI_Polymorphism_Detection_Sim` repository. The project provides an educational and defensive research prototype simulating how polymorphic Windows PE variants evade signature-based detection and how a hybrid machine-learning and heuristic ensemble identifies them.

All critical architectural, mathematical, and security flaws identified in prior iterations—including variant lineage data leakage, synthetic label independence, section desynchronization bugs, preprocessing ordering errors, dashboard vulnerabilities, and code duplication—have been addressed. A full test suite (22 automated tests) was introduced, achieving 100% pass verification across the synthetic generation, feature extraction, group-aware training, model inference, and web dashboard tiers.

---

## 2. Audit Findings & Root-Cause Analysis

### Finding 1: Variant Lineage Data Leakage & Metric Inflation (Critical)
* **Root Cause:** In the original `detection/train_model.py`, the training script used standard row-level stratified splitting:
  ```python
  # Flawed implementation:
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
  ```
* **Impact:** Because variants in `dataset/variants/` are derived via minor perturbations of parent base artifacts, variants sharing the identical root ancestor (`origin_id`) appeared simultaneously in both the training partition and the test partition. Analysis revealed that **100% of variant families in the test set (105 out of 105) overlapped with families in the training set**.
* **Consequence:** The model memorized parent-specific feature signatures rather than learning invariant polymorphic detection principles. This produced an artificially inflated test accuracy of **99.87%**, concealing the model's inability to generalize to genuinely novel, unseen polymorphic variant families.

### Finding 2: Synthetic Label Independence (High)
* **Root Cause:** In `artifact_simulator/simulator.py`, the base generator drew numeric feature fields from arbitrary Gaussian distributions and assigned labels independently using random sampling weights:
  ```python
  # Flawed implementation:
  label = random.choices(["benign", "malware_sim"], weights=[0.6, 0.4], k=1)[0]
  ```
* **Impact:** There was zero mathematical or structural correlation between the generated PE feature vector and the assigned ground-truth label. Features for "malware_sim" were indistinguishable from "benign" in distribution, causing models to learn noise. When evaluated against held-out families under group-aware splitting, accuracy predictably collapsed to chance (~59%).

### Finding 3: Section Desynchronization Bug in Variant Synthesis (Medium)
* **Root Cause:** In `variant_synthesizer/synthesize_variants.py`, `_split_or_shuffle_sizes` split section sizes into two elements when mutating:
  ```python
  # Flawed implementation:
  sizes[idx] = a
  sizes.insert(idx + 1, b)
  # Section entropy was not updated!
  ```
* **Impact:** `section_sizes` increased by one element, but `section_entropy` retained the original length. This caused structural misalignment (`len(section_sizes) != len(section_entropy)`), leading to corrupted variant JSON records.

### Finding 4: Redundant Hardcoded Script Execution (Low / Usability)
* **Root Cause:** In `artifact_simulator/simulator.py` and `variant_synthesizer/synthesize_variants.py`, the entrypoint executed Click CLI and then immediately executed hardcoded demo functions:
  ```python
  if __name__ == "__main__":
      cli(standalone_mode=False)
      generate_base_artifacts(5, ...)  # Unconditionally ran after CLI
  ```
* **Impact:** Any CLI execution generated extra hardcoded files, corrupting test output and intended user arguments.

### Finding 5: Flask Dashboard Security & Unvalidated Inputs (High / Security)
* **Root Cause:** `dashboard/app.py` had no upload size limits, no MIME or extension validation, lacked HTTP security headers, executed with hardcoded `debug=True`, and parsed unvalidated slider payloads directly:
  ```python
  features = request.get_json(force=True)  # Susceptible to crashes on non-JSON
  ```
* **Impact:** Susceptible to Denial of Service (DoS) via oversized uploads or infinite numbers, unhandled 500 exceptions, and cross-origin security exposures.

### Finding 6: Code Duplication Across Extraction & Inference Layers (Medium / Maintainability)
* **Root Cause:** `_flatten_features` and inference logic were independently copy-pasted across `dataset/build_dataset.py`, `detection/infer.py`, and `dashboard/app.py`.
* **Impact:** Changes to feature scaling or extraction in one script led to silent discrepancies and prediction drift across the CLI and dashboard.

---

## 3. Implemented Architecture & Remediations

### 3.1. Unified Detection & Extraction Engine (`detection/detector.py`)
Created a centralized, thread-safe detection module providing:
1. `FEATURE_COLUMNS`: Standardized 7-feature vector:
   `["num_sections", "avg_section_entropy", "total_section_size", "imported_api_count", "strings_count", "junk_bytes_count", "resource_size"]`
2. `extract_features_from_dict(rec)`: Fault-tolerant extraction with NaN handling, missing-field fallbacks, clean type coercion, and lineage resolution.
3. `evaluate_heuristic(features, junk_threshold, entropy_threshold)`: Configurable heuristic evaluator.
4. `PolymorphicDetector`: Reusable, thread-safe class encapsulating scikit-learn pipeline inference, score extraction, heuristic evaluation, and explainability dictionaries.

### 3.2. Class-Conditional Synthetic Simulator (`artifact_simulator/simulator.py`)
Re-engineered the generator to simulate realistic PE structural indicators:
* **Benign Executables:** Characterized by standard section counts (3–6), moderate entropy (mean ~4.1, std ~0.8), minimal junk code (mean ~250 bytes), high standard API imports (30–150), and normal string counts (300–1800).
* **Malware Sim (Packed/Polymorphic):** Characterized by elevated section entropy (mean ~6.85, max up to 7.98 reflecting encrypted payloads/packers), extensive junk byte injection (mean ~4200 bytes, simulating dead code/NOP sleds for signature evasion), reduced direct API imports (5–45, simulating dynamic resolution via `LoadLibrary`/`GetProcAddress`), and suppressed string counts (40–350, simulating encrypted string tables).
* **Lineage & Reproducibility:** Base artifacts explicitly record `origin_id = id`, `generation = 0`, and accept `--seed` for reproducible generation.

### 3.3. Lineage-Preserving Variant Synthesizer (`variant_synthesizer/synthesize_variants.py`)
* **Synchronized Section Mutation:** When a section size is split, section entropy is synchronously split and perturbed with bounded Gaussian noise, strictly enforcing `len(section_sizes) == len(section_entropy) == num_sections`.
* **Lineage Tracking:** Variants record root `origin_id`, immediate `parent_id`, incremented `generation`, and a list of applied `mutations` (`"section_split"`, `"entropy_noise"`, `"junk_mutation"`, `"api_perturbation"`).
* **Label Integrity:** Variants strictly inherit their parent's label.

### 3.4. Leakage-Free Group-Aware Training Pipeline (`detection/train_model.py`)
* **Group-Aware Splitting:** Uses `GroupShuffleSplit` keyed on `origin_id`. Entire variant families are placed exclusively in either the training set or the test set. Zero lineage overlap between partitions is programmatically verified and asserted before fitting.
* **Leakage-Free Preprocessing:** Standard scaling and preprocessing are bundled inside an `sklearn.pipeline.Pipeline`, fitted exclusively on the training partition.
* **Tri-Component Evaluation:** Evaluates and reports metrics independently for:
  1. ML Pipeline Only (Random Forest)
  2. Heuristic Rule Only (`junk_bytes_count > 2500` AND `avg_section_entropy > 6.0`)
  3. Hybrid Ensemble (`max(ML, Heuristic)`)
* **Cross-Validation:** Employs `GroupKFold` across lineage families on the training set to evaluate stability on unseen variants.

### 3.5. Web Dashboard Hardening & Input Validation (`dashboard/app.py`)
* **DoS Protection:** Enforces `app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024` (2 MB).
* **Defense-in-Depth Headers:** Added `X-Content-Type-Options: nosniff`, `X-Frame-Options: SAMEORIGIN`, and strict `Content-Security-Policy`.
* **Strict Validation on `/api/predict`:** Rejects non-JSON bodies, validates numeric bounds for all 7 features, rejects `NaN`/`Infinity`/negative values, and returns clean HTTP 400 JSON errors.
* **Secure File Uploads on `/classify`:** Restricts uploads to `.json` files, safely parses JSON with schema validation, and surfaces user-friendly error banners without exposing server stack traces.
* **Thread-Safe Chart Generation:** Protected Matplotlib figure rendering with thread locking (`threading.Lock()`).
* **Environment Configuration:** Reads `FLASK_DEBUG` from environment; defaults to safe production mode (`debug=False`).

---

## 4. Verification & Automated Test Results

A full automated test suite was implemented in `tests/` and executed using `pytest`:

```
============================= test session starts =============================
platform win32 -- Python 3.10.0, pytest-8.3.5
rootdir: C:\Aadhi Kabilan\Projects\Polymorphic Malware AI Detection\To Github\AI_Polymorphism_Detection_Sim
collected 22 items

tests/test_build_dataset.py::test_build_dataset PASSED                   [  4%]
tests/test_dashboard.py::test_home_route PASSED                          [  9%]
tests/test_dashboard.py::test_metrics_route PASSED                       [ 13%]
tests/test_dashboard.py::test_metrics_charts PASSED                      [ 18%]
tests/test_dashboard.py::test_playground_route PASSED                    [ 22%]
tests/test_dashboard.py::test_dataset_route PASSED                       [ 27%]
tests/test_dashboard.py::test_api_predict_valid PASSED                   [ 31%]
tests/test_dashboard.py::test_api_predict_validation_errors PASSED       [ 36%]
tests/test_dashboard.py::test_classify_upload PASSED                     [ 40%]
tests/test_detector.py::test_extract_features_robustness PASSED          [ 45%]
tests/test_detector.py::test_evaluate_heuristic PASSED                   [ 50%]
tests/test_detector.py::test_zero_group_leakage_guarantee PASSED         [ 54%]
tests/test_detector.py::test_detector_inference PASSED                   [ 59%]
tests/test_detector.py::test_metrics_integrity PASSED                    [ 63%]
tests/test_infer.py::test_classify_artifact PASSED                       [ 68%]
tests/test_infer.py::test_classify_artifact_missing_file PASSED          [ 72%]
tests/test_simulator.py::test_generate_single_artifact_structure PASSED  [ 77%]
tests/test_simulator.py::test_class_conditional_distributions PASSED     [ 81%]
tests/test_simulator.py::test_seed_reproducibility PASSED                [ 86%]
tests/test_synthesizer.py::test_perturb_artifact_lineage_and_alignment PASSED [ 90%]
tests/test_synthesizer.py::test_multi_generation_lineage PASSED          [ 95%]
tests/test_synthesizer.py::test_synthesize_variants_cli PASSED           [100%]

============================= 22 passed in 6.64s ==============================
```

### End-to-End Pipeline Verification
1. **Simulation:** Generated 20 base artifacts with `--seed 42`.
2. **Synthesis:** Created 60 variants (3 per artifact) with `--seed 42`.
3. **Dataset Build:** Generated combined CSV with 80 rows, verifying 20 distinct families with non-null `origin_id`.
4. **Model Training:** Fit group-aware pipeline holding out 4 families (16 samples). Verified 0 family overlap between train and test.
5. **Inference:** Verified single-artifact classification via `detection/infer.py` with structured JSON output.
6. **Dashboard Execution:** Verified all endpoints (`/`, `/metrics`, `/metrics/*.png`, `/classify`, `/playground`, `/dataset`, `/api/predict`).

---

## 5. Data Sources & Provenance Documentation

In accordance with scientific integrity standards, data sources are explicitly documented without inventing benchmark scores or sample counts:

| Dataset Identifier | Nature of Data | Schema / Features | Role in Project | Distribution Status |
| :--- | :--- | :--- | :--- | :--- |
| **Synthetic PE Feature Records** | Synthetic JSON records & CSV | 7 PE structural indicators + IDs | Primary simulation and demonstration pipeline | Publicly checked into GitHub repository |
| **Microsoft Malware Dataset** | External malware sample corpus | Binary samples / PE features | Utilized in broader research by author | Private / restricted; not committed to public repo |
| **Author-Generated Sample Variants** | Feature datasets from sample variants | Extracted PE feature vectors | Utilized in author's experimental research | Private research manifest; separate from synthetic records |

> **Official Research Provenance Statement:**  
> The project combines synthetic feature simulation with an author-reported Microsoft malware dataset and feature records extracted from author-generated sample variants. The public GitHub repository documents the synthetic JSON feature simulation pipeline and provides a safe, reproducible sandbox. Checked-in synthetic datasets are not extracted from Microsoft malware binaries and are not represented as real-world malware samples.

---

## 6. Remaining Limitations & Engineering Roadmap

1. **Synthetic vs. Real Binary Emulation:** The synthetic records simulate PE structural indicators (entropy, section sizes, import counts) as educational proxies. Real-world binary analysis requires deep static disassembly (e.g. IDA, Ghidra, pefile) and dynamic sandbox execution (e.g. CAPEv2, Cuckoo).
2. **Heuristic Rule Calibration:** The current heuristic employs fixed thresholds (junk > 2500 bytes and entropy > 6.0). In production environments, thresholds should be dynamically calibrated against receiver operating characteristic (ROC) curves based on operational false positive tolerances.
3. **Multi-Class Taxonomy:** The current simulation performs binary detection (`benign` vs `malware_sim`). Expanding to multi-class classification (e.g., Ransomware, Trojan, Dropper, Worm) would provide family-level attribution.
4. **Adversarial Hardening:** Future work should incorporate adversarial perturbation training where simulated malware variants actively optimize features to fall just below detection thresholds.

