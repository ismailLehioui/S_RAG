import json
from pathlib import Path

DATASET_PATH = Path(__file__).parent.parent / ".." / "data" / "dataset_orange.json"


def load_orange_toolbox_dataset():
    """Charge le dataset des toolboxes Orange pour le pipeline RAG."""
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data
