### AI_Polymorphism_Detection_Sim

An educational research prototype that demonstrates, using synthetic JSON artifacts only, how polymorphic variants can evade naive signature detection and how an AI+heuristic ensemble can defend. The project does not create, modify, compile, or execute any real binaries. All data are synthetic feature records.

- **No real binaries** are handled.
- **No compilers or binary rewriting libraries** are invoked.
- **All artifacts** are JSON files containing PE-like feature fields.

### Quickstart

1) Create base artifacts (synthetic):
```bash
python artifact_simulator/simulator.py --count 20 --out artifact_simulator/sample_artifacts
```

2) Synthesize polymorphic variants:
```bash
python variant_synthesizer/synthesize_variants.py --in artifact_simulator/sample_artifacts --out dataset/variants --per 30
```

3) Build combined dataset CSV:
```bash
python dataset/build_dataset.py --in artifact_simulator/sample_artifacts --variants dataset/variants --out dataset/combined_features.csv
```

4) Train model and save metrics:
```bash
python detection/train_model.py --data dataset/combined_features.csv --out detection/model
```

5) Run Flask dashboard (dev mode):
```bash
python dashboard/app.py
# or: set FLASK_APP=dashboard/app.py and run: flask run
```
Open your browser at `http://127.0.0.1:5000/`.

6) Classify a single artifact (JSON):
```bash
python detection/infer.py artifact_simulator/sample_artifacts/your_artifact.json
```

### Enhanced Dashboard Features
- **Home**: Overview, ethics, and quick links.
- **Metrics**: KPIs (accuracy/precision/recall), feature importances PNG, confusion matrix PNG, and raw `metrics.json`.
- **Classify**: Upload a JSON artifact to get prediction and explanation.
- **Playground**: Interactive sliders for key features (`num_sections`, `avg_section_entropy`, `total_section_size`, `imported_api_count`, `strings_count`, `junk_bytes_count`, `resource_size`) with AJAX calls to `/api/predict`.
- **Dataset**: Preview first rows of `dataset/combined_features.csv` and label distribution.

#### API
- `POST /api/predict` — JSON body with numeric fields from the Playground; returns model score, label, and explanation.

### Implementation Order (after changes)
1. Install deps:
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1  # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```
2. Generate artifacts and variants:
```bash
python artifact_simulator/simulator.py --count 20
python variant_synthesizer/synthesize_variants.py --in artifact_simulator/sample_artifacts --out dataset/variants --per 30
```
3. Build dataset and train model:
```bash
python dataset/build_dataset.py --in artifact_simulator/sample_artifacts --variants dataset/variants --out dataset/combined_features.csv
python detection/train_model.py --data dataset/combined_features.csv --out detection/model
```
4. Launch the dashboard:
```bash
python dashboard/app.py
# visit http://127.0.0.1:5000/
```
5. Try the new UI:
- Go to `/metrics` to see charts.
- Use `/classify` to upload a JSON.
- Explore `/playground` to see how predictions shift as you adjust features.
- Browse `/dataset` to preview rows and label distribution.

### Ethics and Safety
- **Purpose**: educational and defensive research demonstration.
- **No malicious code** is included.
- **No binary obfuscation or modification tools** are included.
- **All data are synthetic**, stored as JSON or CSV.
- **Air-gapped/sandboxed testing** only if you later extend this to real binaries.

End user note: This project is purely synthetic and educational. If you later want to analyze real binaries for defensive research, first obtain written approval from your institution and follow strict lab/sandboxing best practices (do not run malware on host machines or open networks).

### Troubleshooting
- If charts are empty, ensure the model was trained and `detection/model/metrics.json` exists.
- If `/dataset` says file not found, run dataset build first.
- The Playground requires the trained model to return predictions.
"# AI_Polymorphism_Detection_Sim" 
