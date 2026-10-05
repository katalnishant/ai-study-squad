"""Convert a v1 ``study_logs.json`` into the v2 JSONL log, dropping junk entries.

Usage:
    python scripts/import_legacy_logs.py path/to/old/study_logs.json

Entries that are terminal commands, 'exit', or contain paths/API-key text are
skipped. Review data/study_logs.jsonl afterwards before sharing it anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from study_squad.config import load_settings  # noqa: E402
from study_squad.mining import clean_queries  # noqa: E402
from study_squad.router import detect_intent  # noqa: E402
from study_squad.storage import LogStore, read_records  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 1
    records = read_records(argv[1])
    keep = set(clean_queries([r["query"] for r in records]))
    store = LogStore(load_settings().log_path)
    imported = 0
    for record in records:
        if record["query"] not in keep:
            continue
        route = detect_intent(record["query"])
        record.update(session_id="legacy", intent=route.intent, lead_agent=route.lead)
        store.append(record)
        imported += 1
    print(f"Imported {imported} of {len(records)} records into {store.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
