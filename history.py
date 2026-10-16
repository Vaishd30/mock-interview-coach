import json
from pathlib import Path

HISTORY_FILE = Path(__file__).parent / "interview_history.json"


def load_history():
    """Return saved interviews, newest first."""

    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, list) else []

    except (json.JSONDecodeError, OSError):
        return []


def save_interview(record):
    """Save a new interview record."""

    history = load_history()
    history.insert(0, record)

    with open(HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(
            history,
            file,
            indent=2,
            ensure_ascii=False,
        )


def delete_history():
    """Delete all saved interview records."""

    if HISTORY_FILE.exists():
        HISTORY_FILE.unlink()