from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW, DATA_INTERIM, DATA_PROCESSED = (ROOT / "data" / x for x in ("raw", "interim", "processed"))
MODELS = ROOT / "models"
OUTPUTS, METRICS, PREDICTIONS, CHARTS, TABLES = (ROOT / "outputs" / x for x in ("", "metrics", "predictions", "charts", "tables"))
REPORT, FIGURES = ROOT / "report", ROOT / "report" / "figures"
SEED = 42
# Freddie Mac's yearly sample release uses sample_orig_YYYY.txt and
# sample_svcg_YYYY.txt. Some downloaded copies use sample_perf_YYYY.txt.
# Standard quarterly releases have a different historical_data_* naming scheme.
ORIGINATION_GLOB = "sample_orig*.txt"
PERFORMANCE_GLOBS = ("sample_svcg*.txt", "sample_perf*.txt")
TARGET_MONTHS = 24

def ensure_directories():
    for path in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, MODELS, METRICS, PREDICTIONS, CHARTS, TABLES, REPORT, FIGURES):
        path.mkdir(parents=True, exist_ok=True)
