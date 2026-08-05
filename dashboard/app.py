import os
import io
import json
from flask import Flask, request, jsonify, send_file, render_template_string
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Safety: Web app only interacts with local JSON/CSV and sklearn model.

app = Flask(__name__)

ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(ROOT, 'detection', 'model')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')
MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest.pkl')
DATASET_CSV = os.path.join(ROOT, 'dataset', 'combined_features.csv')


BASE_LAYOUT = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{{ title or 'AI Polymorphism Detection Sim' }}</title>
<style>
  :root { --bg:#0e1217; --card:#151a22; --muted:#9aa4b2; --text:#e6edf3; --accent:#4f8cff; --good:#2ecc71; --bad:#e74c3c; }
  body { margin:0; background:var(--bg); color:var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Fira Sans', 'Droid Sans', 'Helvetica Neue', sans-serif; }
  header { position:sticky; top:0; background:#0f141b; border-bottom:1px solid #1f2630; padding:12px 20px; display:flex; align-items:center; gap:20px; z-index:10; }
  header .brand { font-weight:700; letter-spacing:0.3px; }
  header nav a { color:var(--muted); text-decoration:none; margin-right:16px; }
  header nav a:hover { color:var(--text); }
  .container { max-width:1100px; margin:24px auto; padding:0 16px; }
  .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:16px; }
  .card { background:var(--card); border:1px solid #1f2630; border-radius:12px; padding:16px; }
  .btn { background:var(--accent); color:#fff; border:none; padding:10px 14px; border-radius:8px; cursor:pointer; }
  .btn.secondary { background:#2d3748; }
  .muted { color:var(--muted); }
  input[type=file], input[type=number], input[type=range] { width:100%; margin:8px 0; }
  .kpi { display:flex; gap:16px; }
  .kpi .pill { background:#1b2330; padding:10px 12px; border-radius:8px; border:1px solid #253042; }
  pre { white-space:pre-wrap; word-wrap:break-word; }
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


def load_metrics():
    if os.path.isfile(METRICS_PATH):
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def load_model_bundle():
    if os.path.isfile(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


@app.route('/')
def home():
    body = """
    <div class="grid">
      <div class="card">
        <h2>Overview</h2>
        <p>This educational prototype uses <b>synthetic JSON artifacts</b> to demonstrate how polymorphic variants can evade naive signatures and how an AI+heuristic ensemble can help detect them.</p>
        <p class="muted">No real binaries are created, modified, compiled, or executed. All data are synthetic JSON feature records and CSV datasets.</p>
      </div>
      <div class="card">
        <h2>Quick Links</h2>
        <p><a class="btn" href="/metrics">View Metrics</a></p>
        <p><a class="btn" href="/classify">Classify Artifact (Upload)</a></p>
        <p><a class="btn" href="/playground">Interactive Playground</a></p>
        <p><a class="btn" href="/dataset">Browse Dataset</a></p>
      </div>
      <div class="card">
        <h2>Ethics and Safety</h2>
        <p><b>No real binaries</b> are handled. If you later want to analyze real binaries for defensive research, first obtain written approval from your institution and follow strict lab/sandboxing best practices (do not run malware on host machines or open networks).</p>
      </div>
    </div>
    """
    return page("Home", body)


@app.route('/metrics')
def metrics():
    m = load_metrics()
    kpi = """
      <div class=\"kpi\">
        <div class=\"pill\">Accuracy: <b>{acc}</b></div>
        <div class=\"pill\">Precision: <b>{prec}</b></div>
        <div class=\"pill\">Recall: <b>{rec}</b></div>
      </div>
    """.format(acc=round(m.get('accuracy', 0),4) if m else 'N/A',
                prec=round(m.get('precision', 0),4) if m else 'N/A',
                rec=round(m.get('recall', 0),4) if m else 'N/A')
    body = f"""
    <div class="card">
      <h2>Model Metrics</h2>
      {kpi}
      <h3>Feature Importances</h3>
      <img src="/metrics/feature_importance.png" alt="Feature Importance"/>
      <h3>Confusion Matrix</h3>
      <img src="/metrics/confusion_matrix.png" alt="Confusion Matrix"/>
      <h3>Raw JSON</h3>
      <pre>{json.dumps(m, indent=2) if m else 'No metrics found. Train the model first.'}</pre>
    </div>
    """
    return page("Metrics", body)


@app.route('/metrics/feature_importance.png')
def feature_importance_png():
    m = load_metrics()
    fig, ax = plt.subplots(figsize=(6, 4))
    if m and m.get('feature_cols') and m.get('feature_importances'):
        cols = m['feature_cols']
        imps = m['feature_importances']
        y_pos = np.arange(len(cols))
        ax.barh(y_pos, imps, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cols)
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importances')
        plt.tight_layout()
    else:
        ax.text(0.5, 0.5, 'Train model to view importances', ha='center', va='center')
        ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/metrics/confusion_matrix.png')
def confusion_matrix_png():
    m = load_metrics()
    fig, ax = plt.subplots(figsize=(4, 4))
    if m and m.get('confusion_matrix'):
        cm = np.array(m['confusion_matrix'])
        im = ax.imshow(cm, cmap='Blues')
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, str(z), ha='center', va='center', color='black')
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks([0,1])
        ax.set_yticks([0,1])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
    else:
        ax.text(0.5, 0.5, 'Train model to view confusion matrix', ha='center', va='center')
        ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    result = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            try:
                rec = json.load(file)
                result = _predict_from_record(rec)
            except Exception as e:
                result = {'error': str(e)}
        else:
            result = {'error': 'No file uploaded'}
    body = render_template_string("""
    <div class="card">
      <h2>Classify Artifact (Upload JSON)</h2>
      <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="application/json" required>
        <button class="btn" type="submit">Classify</button>
      </form>
      {% if result %}
      <h3>Result</h3>
      <pre>{{ result | tojson(indent=2) }}</pre>
      {% endif %}
    </div>
    """, result=result)
    return page("Classify", body)


@app.route('/playground')
def playground():
    # Interactive sliders and instant predict via AJAX
    body = """
    <div class="card">
      <h2>Interactive Playground</h2>
      <p class="muted">Adjust features to see prediction changes. This simulates polymorphic feature shifts.</p>
      <div class="grid">
        <div>
          <label>num_sections <span id="num_sections_val">5</span></label>
          <input type="range" id="num_sections" min="1" max="15" value="5" />

          <label>avg_section_entropy <span id="entropy_val">4.0</span></label>
          <input type="range" id="avg_section_entropy" min="0" max="8" step="0.1" value="4.0" />

          <label>total_section_size (bytes) <span id="size_val">200000</span></label>
          <input type="range" id="total_section_size" min="0" max="2000000" step="1000" value="200000" />

          <label>imported_api_count <span id="apis_val">50</span></label>
          <input type="range" id="imported_api_count" min="0" max="200" value="50" />

          <label>strings_count <span id="strings_val">500</span></label>
          <input type="range" id="strings_count" min="0" max="3000" value="500" />

          <label>junk_bytes_count <span id="junk_val">2000</span></label>
          <input type="range" id="junk_bytes_count" min="0" max="10000" step="50" value="2000" />

          <label>resource_size (bytes) <span id="res_val">40000</span></label>
          <input type="range" id="resource_size" min="0" max="300000" step="1000" value="40000" />
          <br/>
          <button class="btn" id="predict_btn">Predict</button>
        </div>
        <div>
          <h3>Prediction</h3>
          <pre id="result_box" class="muted">Adjust sliders and click Predict.</pre>
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
        const res = await fetch('/api/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
        const data = await res.json();
        document.getElementById('result_box').textContent = JSON.stringify(data, null, 2);
      });
    </script>
    """
    return page("Playground", body)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        features = request.get_json(force=True)
        # Build a minimal record compatible with predictor
        rec = {
            'section_sizes': [],
            'section_entropy': [],
            'num_sections': int(features.get('num_sections', 5)),
            'imported_api_count': int(features.get('imported_api_count', 50)),
            'strings_count': int(features.get('strings_count', 500)),
            'junk_bytes_count': int(features.get('junk_bytes_count', 2000)),
            'resource_size': int(features.get('resource_size', 40000)),
        }
        # Store aggregated fields directly
        rec['section_entropy'] = [float(features.get('avg_section_entropy', 4.0))] * max(rec['num_sections'], 1)
        # Simulate sizes that sum to total_section_size
        total = int(features.get('total_section_size', 200000))
        if rec['num_sections'] > 0:
            base = total // rec['num_sections']
            sizes = [base] * rec['num_sections']
            remainder = total - base * rec['num_sections']
            for i in range(remainder):
                sizes[i % rec['num_sections']] += 1
            rec['section_sizes'] = sizes
        result = _predict_from_record(rec)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/dataset')
def dataset_browser():
    # Preview top rows of dataset and basic label distribution
    if not os.path.isfile(DATASET_CSV):
        body = """
        <div class="card">
          <h2>Dataset</h2>
          <p>No dataset found. Build it first from the CLI steps in README.</p>
        </div>
        """
        return page("Dataset", body)
    try:
        df = pd.read_csv(DATASET_CSV)
        head = df.head(20).to_dict(orient='records')
        labels = df['label'].value_counts().to_dict()
        body = render_template_string("""
        <div class="card">
          <h2>Dataset Preview</h2>
          <p class="muted">First 20 rows from dataset/combined_features.csv</p>
          <pre>{{ head | tojson(indent=2) }}</pre>
          <h3>Label Distribution</h3>
          <pre>{{ labels | tojson(indent=2) }}</pre>
        </div>
        """, head=head, labels=labels)
        return page("Dataset", body)
    except Exception as e:
        return page("Dataset", f"<div class=card><h2>Dataset Error</h2><pre>{str(e)}</pre></div>")


def _predict_from_record(rec: dict) -> dict:
    bundle = load_model_bundle()
    if bundle is None:
        return {"error": "Model not found. Train first."}
    cols = bundle['feature_cols']
    heuristic = bundle.get('heuristic', {"junk_threshold": 2500, "entropy_threshold": 6.0})
    sizes = rec.get('section_sizes', []) or []
    ent = rec.get('section_entropy', []) or []
    num_sections = int(rec.get('num_sections', len(sizes) or len(ent)))
    avg_entropy = float(np.mean(ent)) if len(ent) > 0 else 0.0
    total_size = int(np.sum(sizes)) if len(sizes) > 0 else 0
    flat = {
        'num_sections': num_sections,
        'avg_section_entropy': avg_entropy,
        'total_section_size': total_size,
        'imported_api_count': int(rec.get('imported_api_count', 0)),
        'strings_count': int(rec.get('strings_count', 0)),
        'junk_bytes_count': int(rec.get('junk_bytes_count', 0)),
        'resource_size': int(rec.get('resource_size', 0)),
    }
    X = np.array([[flat[c] for c in cols]], dtype=float)
    score = float(bundle['model'].predict_proba(X)[:, 1][0])
    pred_ml = int(score >= 0.5)
    h_flag = int((flat['junk_bytes_count'] > heuristic.get('junk_threshold', 2500)) and (flat['avg_section_entropy'] > heuristic.get('entropy_threshold', 6.0)))
    pred_final = int(max(pred_ml, h_flag))
    return {
        'score': round(score, 6),
        'pred': 'malware_sim' if pred_final == 1 else 'benign',
        'explanation': {
            'pred_ml': pred_ml,
            'heuristic_flag': h_flag,
            'junk_bytes_count': flat['junk_bytes_count'],
            'avg_section_entropy': round(flat['avg_section_entropy'], 6),
            'imported_api_count': flat['imported_api_count'],
            'strings_count': flat['strings_count'],
            'total_section_size': flat['total_section_size'],
        },
    }


if __name__ == '__main__':
    # Dev server
    app.run(host='127.0.0.1', port=5000, debug=True)
