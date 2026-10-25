from pathlib import Path
from src.data_loader import ORIG_COLS, PERF_COLS, _field_count

def test_2022_sample_layout_counts():
    assert _field_count(Path("data/raw/sample_orig_2022.txt")) == len(ORIG_COLS) == 31
    assert _field_count(Path("data/raw/sample_perf_2022.txt")) == len(PERF_COLS) == 35
