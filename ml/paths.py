"""Where data lives and where outputs go. Override with environment variables.

RUSETECT_DATA  folder with aml_HI-Small_part0*.csv.gz, HI-Small_Patterns.txt, baf_Base_part*.csv.gz
RUSETECT_WORK  folder for intermediate parquet files and predictions
"""
import os


def _env(name, default):
    return os.environ.get(f"RUSETECT_{name}") or default


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = _env("DATA", os.path.join(REPO, "data"))
WORK = _env("WORK", os.path.join(REPO, "work"))
MODELS = os.path.join(REPO, "service", "models")       # artefacts served by the API
EXPORT = os.path.join(REPO, "ml", "exported")          # native LightGBM text models
for _d in (WORK, EXPORT):
    os.makedirs(_d, exist_ok=True)
