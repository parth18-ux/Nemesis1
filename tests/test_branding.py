"""Protect the RuseTect rename, model loading, threshold export and report consistency."""
import ast, base64, json, re, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "service"))
sys.path.insert(0, str(ROOT / "ml"))

from fastapi.testclient import TestClient  # noqa: E402
from index import app  # noqa: E402

client = TestClient(app)
STATIC = ROOT / "service" / "static"
# old project names and old event branding, base64 encoded so this guard does not itself contain them
LEGACY = re.compile(base64.b64decode("cmFrc2hhbXxjaXRhZGVsfGlpdFtcc18tXSpkZWxoaXx0cmFja1xzKjA/MlxifGFtYXpvbg==").decode(), re.I)
REPO_URL = "github.com/parth2581/RuseTect"


# ---------------------------------------------------------------- rename
@pytest.mark.parametrize("path,title", [("/", "<title>RuseTect</title>"), ("/console", "<title>RuseTect Live Console</title>"),
                                        ("/report", "<title>RuseTect Benchmark</title>")])
def test_pages_use_new_name_and_no_legacy_branding(path, title):
    html = client.get(path).text.replace(REPO_URL, "")
    assert title in html
    assert not LEGACY.search(html), LEGACY.search(html).group(0)
    assert 'href="/favicon.svg"' in html


def test_api_identity():
    h = client.get("/api/health").json()
    assert h["name"] == "RuseTect" and h["id"] == "rusetect" and h["engine_loaded"] is True
    assert client.get("/openapi.json").json()["info"]["title"] == "RuseTect API"
    assert client.get("/favicon.svg").status_code == 200


def test_no_legacy_names_anywhere_in_source():
    """No old project names in any source, doc, config or results file."""
    hits = []
    for p in ROOT.rglob("*"):
        if p.is_dir() or any(part in {".git", "__pycache__", ".pytest_cache", ".venv", "work", "data"} for part in p.parts):
            continue
        if p.suffix not in {".py", ".html", ".md", ".json", ".ps1", ".txt", ".svg", ".toml", ".yml", ".yaml", ".cfg", ".ini", ".sh"} and p.name not in {"Dockerfile", ".gitignore", ".vercelignore", "vercel.json"}:
            continue
        rel = p.relative_to(ROOT).as_posix()
        for n, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if LEGACY.search(line):
                hits.append(f"{rel}:{n}: {line.strip()[:80]}")
    assert not hits, hits


# ---------------------------------------------------------------- models + thresholds
def test_engine_loads_all_live_models():
    from server import ENGINE
    assert ENGINE is not None and ENGINE.models_loaded
    for attr in ("l1", "l2x", "l2l", "l0"):
        assert getattr(ENGINE, attr) is not None
    assert not hasattr(ENGINE, "rf") and not hasattr(ENGINE, "lstm")     # offline-only models are never loaded


def test_config_has_every_threshold_the_engine_reads():
    cfg = json.loads((ROOT / "service" / "models" / "cascade_config.json").read_text())
    for k in ("l1_threshold", "block_threshold", "hold_threshold", "verify_threshold"):
        assert isinstance(cfg[k], float), k
    assert cfg["l1_threshold"] < cfg["verify_threshold"] < cfg["hold_threshold"] < cfg["block_threshold"]


def test_cascade_exporter_writes_verify_threshold():
    src = (ROOT / "ml" / "cascade.py").read_text()
    tree = ast.parse(src)
    exported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "dict":
            exported |= {kw.arg for kw in node.keywords}
    for k in ("l1_threshold", "block_threshold", "hold_threshold", "verify_threshold", "features", "fmt_codes", "cur_codes", "usd_rates"):
        assert k in exported, k
    assert "VERIFY_THRESHOLD = 0.10" in src     # must equal the committed config so a retrain keeps decisions


def test_engine_rejects_config_without_thresholds(tmp_path):
    import shutil
    from rusetect_engine import Engine
    for f in (ROOT / "service" / "models").iterdir():
        shutil.copy(f, tmp_path / f.name)
    cfg = json.loads((tmp_path / "cascade_config.json").read_text()); cfg.pop("verify_threshold")
    (tmp_path / "cascade_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="verify_threshold"):
        Engine(tmp_path, load_models=False)


# ---------------------------------------------------------------- report
def test_built_report_is_up_to_date():
    import build_report
    assert (STATIC / "report.html").read_text() == build_report.build(), "run: python ml/build_report.py"


def test_report_embeds_committed_benchmark_exactly():
    html = (STATIC / "report.html").read_text()
    data = json.loads(re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1))
    bench = json.loads((ROOT / "service" / "models" / "benchmark_results.json").read_text())
    assert data["aml"] == bench["aml_models"] and data["baf"] == bench["baf_models"]
    assert data["cascade"] == bench["cascade"] and data["l0"] == bench["l0_final"]


def test_landing_page_benchmark_numbers_match_committed_results():
    html = (STATIC / "index.html").read_text()
    bench = json.loads((ROOT / "service" / "models" / "benchmark_results.json").read_text())["aml_models"]
    shown = dict(re.findall(r'\["([^"]+)", ([0-9.]+)', html))
    mapping = {"XGBoost + graph features": "XGBoost+graph", "Random Forest + graph": "RandomForest+graph", "CatBoost + graph": "CatBoost+graph",
               "LightGBM + graph": "LightGBM+graph", "Decision Tree + graph": "DecisionTree+graph", "Logistic Regression + graph": "LogisticRegression+graph",
               "MLP": "MLP", "XGBoost, raw fields only": "XGBoost"}
    for label, key in mapping.items():
        assert label in shown, label
        assert abs(float(shown[label]) - bench[key]["pr_auc"]) < 0.0005, label


# ---------------------------------------------------------------- LightGBM served without the lightgbm package
def test_lite_booster_matches_native_lightgbm():
    lgb = pytest.importorskip("lightgbm")
    import numpy as np
    from lgbm_lite import LiteBooster
    rng = np.random.default_rng(0)
    for name in ("l2_lgbm", "l0_onboarding_lgbm"):
        native = lgb.Booster(model_file=str(ROOT / "ml" / "exported" / f"{name}.txt"))
        lite = LiteBooster(ROOT / "service" / "models" / f"{name}.json")
        X = rng.normal(0, 3, size=(300, native.num_feature()))
        X[rng.random(X.shape) < 0.1] = np.nan                       # exercise missing-value routing
        if name.startswith("l0"):
            X = np.abs(np.round(X))                                  # categorical columns take small non-negative codes
        assert np.abs(native.predict(X) - lite.predict(X)).max() < 1e-9, name
