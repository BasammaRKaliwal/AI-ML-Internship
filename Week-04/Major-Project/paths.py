"""File and folder locations (no PyTorch needed)."""
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"
IMAGES_DIR = BASE / "images"
FINAL_MODEL = BASE / "final_model.pt"
SPLIT_INFO = BASE / "split_info.json"
HISTORY_FILE = BASE / "history.json"
RESULTS_FILE = BASE / "results.json"
REPORT_FILE = BASE / "evaluation_report.md"
COMPARISON_FILE = BASE / "comparison_table.md"
