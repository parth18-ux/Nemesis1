"""Where data lives and where outputs go. Override with environment variables."""
import os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("RUSETECT_DATA", os.path.join(REPO, "data"))          # IBM HI-Small + BAF csv(.gz) chunks
WORK = os.environ.get("RUSETECT_WORK", os.path.join(REPO, "work"))          # intermediate parquet / predictions
MODELS = os.path.join(REPO, "service", "models")                               # artefacts served by the API
EXPORT = os.path.join(REPO, "ml", "exported")                              # native LightGBM text models
for _d in (WORK, EXPORT):
    os.makedirs(_d, exist_ok=True)
