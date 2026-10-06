"""Central configuration for the analytics pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
MODELS, OUT, CHARTS, REPORTS = ROOT / "models", ROOT / "outputs", ROOT / "outputs" / "charts", ROOT / "reports"

N_STORES = 40
START, END = "2020-01-01", "2023-12-31"
TEST_START = "2023-01-01"          # time-based split: train < 2023, test = 2023
HORIZON = 28                        # forecast horizon in days (also the minimum lag -> no leakage)
SEED = 42
for p in (RAW, PROC, MODELS, OUT, CHARTS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)
