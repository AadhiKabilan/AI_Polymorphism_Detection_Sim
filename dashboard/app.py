import os
import io
import sys
import json
import threading
from typing import Optional, Dict, Any

from flask import Flask, request, jsonify, send_file, render_template_string
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from detection.detector import PolymorphicDetector, FEATURE_COLUMNS

# Safety: Web application strictly interacts with local synthetic JSON/CSV files and sklearn model.
# No binary execution or generation.

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB upload limit to prevent DoS
app.config["MODEL_DIR"] = os.path.join(REPO_ROOT, "detection", "model")
app.config["METRICS_PATH"] = os.path.join(app.config["MODEL_DIR"], "metrics.json")
app.config["MODEL_PATH"] = os.path.join(app.config["MODEL_DIR"], "random_forest.pkl")
app.config["DATASET_CSV"] = os.path.join(REPO_ROOT, "dataset", "combined_features.csv")

# Plot rendering lock for thread safety with Agg backend
plot_lock = threading.Lock()


@app.after_request
def apply_security_headers(response):
    """Add defense-in-depth security headers."""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:;"
    )
    return response


BASE_LAYOUT = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{{ title or 'AI Polymorphism Detection Sim' }}</title>
<style>
  :root { --bg:#0e1217; --card:#151a22; --muted:#9aa4b2; --text:#e6edf3; --accent:#4f8cff; --good:#2ecc71; --bad:#e74c3c; --border:#1f2630; }
  body { margin:0; background:var(--bg); color:var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; }
  header { position:sticky; top:0; background:#0f141b; border-bottom:1px solid var(--border); padding:14px 24px; display:flex; align-items:center; justify-content:space-between; z-index:10; }
  header .brand { font-weight:700; font-size:1.1rem; letter-spacing:0.3px; color:#fff; }
  header nav a { color:var(--muted); text-decoration:none; margin-left:20px; font-weight:500; font-size:0.95rem; }
  header nav a:hover { color:var(--text); }
  .container { max-width:1100px; margin:28px auto; padding:0 20px; }
  .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap:20px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px; margin-bottom:20px; }
  .btn { background:var(--accent); color:#fff; border:none; padding:10px 16px; border-radius:8px; cursor:pointer; font-weight:600; text-decoration:none; display:inline-block; }
  .btn:hover { opacity:0.9; }
  .muted { color:var(--muted); }
  .badge { display:inline-block; padding:4px 10px; border-radius:6px; font-size:0.85rem; font-weight:600; }
  .badge.good { background:rgba(46, 204, 113, 0.15); color:var(--good); border:1px solid var(--good); }
  .badge.bad { background:rgba(231, 76, 60, 0.15); color:var(--bad); border:1px solid var(--bad); }
  .badge.info { background:rgba(79, 140, 255, 0.15); color:var(--accent); border:1px solid var(--accent); }
  input[type=file], input[type=number], input[type=range] { width:100%; box-sizing:border-box; margin:8px 0; }
  .kpi-row { display:flex; flex-wrap:wrap; gap:16px; margin:16px 0; }
  .kpi-pill { background:#1b2330; padding:14px 18px; border-radius:10px; border:1px solid #253042; flex:1; min-width:140px; }
  .kpi-pill .val { font-size:1.4rem; font-weight:700; margin-top:4px; }
  pre { background:#0a0d12; border:1px solid var(--border); border-radius:8px; padding:14px; overflow-x:auto; font-size:0.88rem; }
  .alert { padding:12px 16px; border-radius:8px; margin-bottom:16px; }
  .alert.error { background:rgba(231, 76, 60, 0.15); border:1px solid var(--bad); color:#ff8080; }
  .alert.success { background:rgba(46, 204, 113, 0.15); border:1px solid var(--good); color:#80ffaa; }
  table { width:100%; border-collapse:collapse; margin-top:12px; font-size:0.88rem; }
  th, td { border:1px solid var(--border); padding:8px 12px; text-align:left; }
  th { background:#10151d; color:var(--muted); }
</style>
</head>
<body>
<header>
  <div class="brand">AI_Polymorphism_Detection_Sim</div>
  <nav>
    <a href="/">Home</a>
    <a href="/metrics">Metrics</a>
    <a href="/classify">Classify</a>
    <a href="/playground">Playground</a>
    <a href="/dataset">Dataset</a>
  </nav>
</header>
<div class="container">
  {{ body|safe }}
</div>
</body>
</html>
"""


def page(title: str, body: str):
    return render_template_string(BASE_LAYOUT, title=title, body=body)


def load_metrics() -> Optional[Dict[str, Any]]:
    path = app.config.get("METRICS_PATH")
    if path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def get_detector() -> Optional[PolymorphicDetector]:
    path = app.config.get("MODEL_PATH")
    if path and os.path.isfile(path):
        try:
            return PolymorphicDetector.load(path)
        except Exception:
            return None
    return None


@app.route("/")
def home():
    body = """
    <div class="grid">
      <div class="card">
        <h2>System Overview</h2>
        <p><b>AI_Polymorphism_Detection_Sim</b> is an educational research simulation demonstrating how polymorphic PE variants attempt to evade naive signature detection and how a machine learning model paired with structural heuristics can detect them.</p>
        <p class="muted">All artifacts and datasets are <b>strictly synthetic JSON feature records and CSVs</b>. No real malware binaries are handled, executed, or generated.</p>
        <span class="badge info">Safe Sandbox Simulation</span>
      </div>
      <div class="card">
        <h2>Quick Actions</h2>
        <p><a class="btn" href="/metrics">View Evaluation Metrics</a></p>
        <p><a class="btn" href="/classify">Classify Artifact (Upload)</a></p>
        <p><a class="btn" href="/playground">Interactive Sliders Playground</a></p>
        <p><a class="btn" href="/dataset">Browse Dataset & Families</a></p>
      </div>
    </div>
    <div class="card">
      <h2>Research Provenance & Integrity</h2>
      <p>This prototype implements <b>group-aware train/test splitting</b> by variant lineage origin. This guarantees that variants derived from the same parent do not span across train and test partitions, eliminating variant lineage data leakage.</p>
      <p class="muted">Author research notes confirm the use of an external Microsoft malware dataset and extracted sample variants. Because those external samples require separate custodial provenance and permission records, this public repository demonstrates the pipeline exclusively on verified synthetic feature distributions.</p>
    </div>
    """
    return page("Home", body)


@app.route("/metrics")
def metrics():
    m = load_metrics()
    if not m:
        body = """
        <div class="card">
          <h2>Evaluation Metrics</h2>
          <div class="alert error">No metrics found. Train the model first using <code>python detection/train_model.py --data dataset/combined_features.csv</code></div>
        </div>
        """
        return page("Metrics", body)

    leakage_badge = (
        '<span class="badge good">Group-Aware Split (Zero Lineage Leakage)</span>'
        if m.get("samples", {}).get("zero_leakage_verified")
        else '<span class="badge bad">Naive Split (Leakage Present)</span>'
    )

    kpi = f"""
    <div class="kpi-row">
      <div class="kpi-pill"><span class="muted">Accuracy (Ensemble)</span><div class="val">{round(m.get('accuracy', 0), 4)}</div></div>
      <div class="kpi-pill"><span class="muted">Precision</span><div class="val">{round(m.get('precision', 0), 4)}</div></div>
      <div class="kpi-pill"><span class="muted">Recall</span><div class="val">{round(m.get('recall', 0), 4)}</div></div>
      <div class="kpi-pill"><span class="muted">F1-Score</span><div class="val">{round(m.get('f1_score', 0), 4)}</div></div>
    </div>
    """

    ml_kpi = m.get("ml_only", {})
    h_kpi = m.get("heuristic_only", {})
    samples = m.get("samples", {})

    body = f"""
    <div class="card">
      <h2>Model Evaluation & Validation</h2>
      <p>{leakage_badge}</p>
      {kpi}

      <h3>Independent Component Comparison</h3>
      <table>
        <tr><th>Mode</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr>
        <tr><td><b>ML Pipeline (Random Forest)</b></td><td>{round(ml_kpi.get('accuracy', 0), 4)}</td><td>{round(ml_kpi.get('precision', 0), 4)}</td><td>{round(ml_kpi.get('recall', 0), 4)}</td><td>{round(ml_kpi.get('f1_score', 0), 4)}</td></tr>
        <tr><td><b>Heuristic Only (Entropy & Junk Rule)</b></td><td>{round(h_kpi.get('accuracy', 0), 4)}</td><td>{round(h_kpi.get('precision', 0), 4)}</td><td>{round(h_kpi.get('recall', 0), 4)}</td><td>{round(h_kpi.get('f1_score', 0), 4)}</td></tr>
        <tr><td><b>Hybrid Ensemble (ML + Heuristic)</b></td><td><b>{round(m.get('accuracy', 0), 4)}</b></td><td><b>{round(m.get('precision', 0), 4)}</b></td><td><b>{round(m.get('recall', 0), 4)}</b></td><td><b>{round(m.get('f1_score', 0), 4)}</b></td></tr>
      </table>

      <p class="muted">Partition summary: {samples.get('train', 'N/A')} train samples ({samples.get('train_families', 'N/A')} families), {samples.get('test', 'N/A')} test samples ({samples.get('test_families', 'N/A')} families).</p>

      <div class="grid" style="margin-top:20px;">
        <div>
          <h3>Feature Importances</h3>
          <img src="/metrics/feature_importance.png" alt="Feature Importance" style="max-width:100%; border-radius:8px; border:1px solid #1f2630;"/>
        </div>
        <div>
          <h3>Confusion Matrix</h3>
          <img src="/metrics/confusion_matrix.png" alt="Confusion Matrix" style="max-width:100%; border-radius:8px; border:1px solid #1f2630;"/>
        </div>
      </div>

      <h3 style="margin-top:24px;">Raw Metrics Metadata</h3>
      <pre>{json.dumps(m, indent=2)}</pre>
    </div>
    """
    return page("Metrics", body)


@app.route("/metrics/feature_importance.png")
def feature_importance_png():
    m = load_metrics()
    with plot_lock:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("#151a22")
        ax.set_facecolor("#151a22")

        if m and m.get("feature_cols") and m.get("feature_importances"):
            cols = m["feature_cols"]
            imps = m["feature_importances"]
            y_pos = np.arange(len(cols))
            ax.barh(y_pos, imps, color="#4f8cff", edgecolor="#253042")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(cols, color="#e6edf3", fontsize=9)
            ax.tick_params(axis="x", colors="#e6edf3")
            ax.spines["bottom"].set_color("#253042")
            ax.spines["left"].set_color("#253042")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_xlabel("Importance", color="#e6edf3")
            ax.set_title("Random Forest Feature Importances", color="#ffffff")
            plt.tight_layout()
        else:
            ax.text(0.5, 0.5, "Train model to view importances", ha="center", va="center", color="#9aa4b2")
            ax.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype="image/png")


@app.route("/metrics/confusion_matrix.png")
def confusion_matrix_png():
    m = load_metrics()
    with plot_lock:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        fig.patch.set_facecolor("#151a22")
        ax.set_facecolor("#151a22")

        if m and m.get("confusion_matrix"):
            cm = np.array(m["confusion_matrix"])
            im = ax.imshow(cm, cmap="Blues")
            for (i, j), z in np.ndenumerate(cm):
                color = "white" if z > (cm.max() / 2.0) else "black"
                ax.text(j, i, str(z), ha="center", va="center", color=color, fontweight="bold")
            ax.set_title("Confusion Matrix (Hold-out Test)", color="#ffffff")
            ax.set_xlabel("Predicted Label", color="#e6edf3")
            ax.set_ylabel("True Ground-Truth", color="#e6edf3")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Benign", "Malware_Sim"], color="#e6edf3")
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Benign", "Malware_Sim"], color="#e6edf3")
            ax.tick_params(colors="#e6edf3")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
        else:
            ax.text(0.5, 0.5, "Train model to view confusion matrix", ha="center", va="center", color="#9aa4b2")
            ax.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return send_file(buf, mimetype="image/png")


@app.route("/classify", methods=["GET", "POST"])
def classify():
    result = None
    error = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            error = "No file selected. Please upload a valid JSON artifact."
        elif not file.filename.lower().endswith(".json"):
            error = "Invalid file type. Only JSON artifacts are accepted."
        else:
            try:
                content = file.read()
                rec = json.loads(content.decode("utf-8"))
                if not isinstance(rec, dict):
                    error = "Invalid JSON structure. Root element must be an object/dictionary."
                else:
                    detector = get_detector()
                    if detector is None:
                        error = "Model not found. Train the model before classifying artifacts."
                    else:
                        result = detector.predict_record(rec)
            except json.JSONDecodeError as e:
                error = f"Malformed JSON file: {e.msg}"
            except Exception as e:
                error = f"Classification error: {str(e)}"

    body = render_template_string("""
    <div class="card">
      <h2>Classify Synthetic Artifact</h2>
      <p class="muted">Upload a synthetic JSON feature record generated by the artifact simulator or variant synthesizer.</p>

      {% if error %}
      <div class="alert error">{{ error }}</div>
      {% endif %}

      <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".json,application/json" required />
        <button class="btn" type="submit" style="margin-top:10px;">Analyze Artifact</button>
      </form>

      {% if result %}
      <div style="margin-top:24px;">
        <h3>Classification Result</h3>
        {% if result.pred == 'malware_sim' %}
          <div class="alert error" style="font-weight:700;">VERDICT: MALWARE_SIM (Score: {{ result.score }})</div>
        {% else %}
          <div class="alert success" style="font-weight:700;">VERDICT: BENIGN (Score: {{ result.score }})</div>
        {% endif %}
        <pre>{{ result | tojson(indent=2) }}</pre>
      </div>
      {% endif %}
    </div>
    """, result=result, error=error)
    return page("Classify", body)


@app.route("/playground")
def playground():
    body = """
    <div class="card">
      <h2>Interactive Polymorphic Feature Playground</h2>
      <p class="muted">Adjust PE structural features below to simulate polymorphic mutations (e.g., junk code injection, section splitting, and entropy shifts). AJAX calls to <code>/api/predict</code> update predictions dynamically.</p>
      <div class="grid">
        <div>
          <label>num_sections: <b id="num_sections_val">5</b></label>
          <input type="range" id="num_sections" min="1" max="15" value="5" />

          <label>avg_section_entropy: <b id="entropy_val">4.0</b></label>
          <input type="range" id="avg_section_entropy" min="0" max="8" step="0.1" value="4.0" />

          <label>total_section_size (bytes): <b id="size_val">200000</b></label>
          <input type="range" id="total_section_size" min="0" max="2000000" step="5000" value="200000" />

          <label>imported_api_count: <b id="apis_val">50</b></label>
          <input type="range" id="imported_api_count" min="0" max="200" value="50" />

          <label>strings_count: <b id="strings_val">500</b></label>
          <input type="range" id="strings_count" min="0" max="3000" step="20" value="500" />

          <label>junk_bytes_count: <b id="junk_val">2000</b></label>
          <input type="range" id="junk_bytes_count" min="0" max="10000" step="100" value="2000" />

          <label>resource_size (bytes): <b id="res_val">40000</b></label>
          <input type="range" id="resource_size" min="0" max="300000" step="2000" value="40000" />
          <br/>
          <button class="btn" id="predict_btn" style="margin-top:12px;">Evaluate Features</button>
        </div>
        <div>
          <h3>Inference Verdict</h3>
          <div id="verdict_banner" style="margin-bottom:12px;"></div>
          <pre id="result_box" class="muted">Adjust sliders and click Evaluate Features.</pre>
        </div>
      </div>
    </div>
    <script>
      function val(id) { return document.getElementById(id).value; }
      function setText(id, v) { document.getElementById(id).innerText = v; }
      const ids = ['num_sections','avg_section_entropy','total_section_size','imported_api_count','strings_count','junk_bytes_count','resource_size'];
      const mirror = {
        'num_sections':'num_sections_val',
        'avg_section_entropy':'entropy_val',
        'total_section_size':'size_val',
        'imported_api_count':'apis_val',
        'strings_count':'strings_val',
        'junk_bytes_count':'junk_val',
        'resource_size':'res_val'
      };
      ids.forEach(id => {
        const el = document.getElementById(id);
        el.addEventListener('input', () => setText(mirror[id], el.value));
      });

      document.getElementById('predict_btn').addEventListener('click', async () => {
        const payload = {};
        ids.forEach(id => payload[id] = parseFloat(val(id)));
        try {
          const res = await fetch('/api/predict', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body: JSON.stringify(payload)
          });
          const data = await res.json();
          document.getElementById('result_box').textContent = JSON.stringify(data, null, 2);
          const banner = document.getElementById('verdict_banner');
          if (data.error) {
            banner.innerHTML = '<div class="alert error">' + data.error + '</div>';
          } else if (data.pred === 'malware_sim') {
            banner.innerHTML = '<div class="alert error" style="font-weight:700;">MALWARE_SIM (Score: ' + data.score + ')</div>';
          } else {
            banner.innerHTML = '<div class="alert success" style="font-weight:700;">BENIGN (Score: ' + data.score + ')</div>';
          }
        } catch (err) {
          document.getElementById('result_box').textContent = 'Error: ' + err;
        }
      });
    </script>
    """
    return page("Playground", body)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Secure endpoint for predicting from raw numeric feature dictionary with strict input validation."""
    if not request.is_json:
        return jsonify({"error": "Request body must be valid application/json"}), 400

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "JSON payload must be an object"}), 400

    # Bounds and type validation
    bounds = {
        "num_sections": (1, 50),
        "avg_section_entropy": (0.0, 8.0),
        "total_section_size": (0, 500_000_000),
        "imported_api_count": (0, 5_000),
        "strings_count": (0, 500_000),
        "junk_bytes_count": (0, 50_000_000),
        "resource_size": (0, 500_000_000),
    }

    clean_features: Dict[str, Any] = {}
    for field, (min_v, max_v) in bounds.items():
        raw = payload.get(field)
        if raw is None:
            # Assign reasonable midpoints or zero
            clean_features[field] = 0.0 if field != "num_sections" else 5
            continue
        try:
            val = float(raw)
            if np.isnan(val) or np.isinf(val):
                return jsonify({"error": f"Invalid numeric value for {field}: {raw}"}), 400
            if val < min_v or val > max_v:
                return jsonify({"error": f"Value for {field} out of allowed bounds [{min_v}, {max_v}]: {val}"}), 400
            clean_features[field] = int(val) if field != "avg_section_entropy" else round(val, 6)
        except (ValueError, TypeError):
            return jsonify({"error": f"Field {field} must be numeric"}), 400

    detector = get_detector()
    if detector is None:
        return jsonify({"error": "Model not loaded. Train the model first."}), 503

    try:
        result = detector.predict_features(clean_features)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Inference failure: {str(e)}"}), 500


@app.route("/dataset")
def dataset_browser():
    dataset_csv = app.config.get("DATASET_CSV")
    if not dataset_csv or not os.path.isfile(dataset_csv):
        body = """
        <div class="card">
          <h2>Dataset Preview</h2>
          <div class="alert error">No dataset found at <code>dataset/combined_features.csv</code>. Build it first via CLI.</div>
        </div>
        """
        return page("Dataset", body)

    try:
        df = pd.read_csv(dataset_csv)
        total_rows = len(df)
        labels = df["label"].value_counts().to_dict() if "label" in df.columns else {}
        families = df["origin_id"].nunique() if "origin_id" in df.columns else "N/A"

        head_records = df.head(15).to_dict(orient="records")

        body = render_template_string("""
        <div class="card">
          <h2>Dataset Overview & Lineage Families</h2>
          <div class="kpi-row">
            <div class="kpi-pill"><span class="muted">Total Records</span><div class="val">{{ total_rows }}</div></div>
            <div class="kpi-pill"><span class="muted">Lineage Families</span><div class="val">{{ families }}</div></div>
            <div class="kpi-pill"><span class="muted">Benign</span><div class="val">{{ labels.get('benign', 0) }}</div></div>
            <div class="kpi-pill"><span class="muted">Malware Sim</span><div class="val">{{ labels.get('malware_sim', 0) }}</div></div>
          </div>

          <h3>First 15 Rows (dataset/combined_features.csv)</h3>
          <pre>{{ head_records | tojson(indent=2) }}</pre>

          <h3>Dataset Schema</h3>
          <table>
            <tr><th>Column</th><th>Dtype</th></tr>
            {% for col, dt in dtypes.items() %}
            <tr><td><b>{{ col }}</b></td><td>{{ dt }}</td></tr>
            {% endfor %}
          </table>
        </div>
        """, total_rows=total_rows, labels=labels, families=families, head_records=head_records, dtypes=df.dtypes.astype(str).to_dict())
        return page("Dataset", body)
    except Exception as e:
        return page("Dataset", f"<div class='card alert error'>Error loading dataset: {str(e)}</div>")


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") in ("1", "true", "True")
    app.run(host="127.0.0.1", port=5000, debug=debug_mode)
