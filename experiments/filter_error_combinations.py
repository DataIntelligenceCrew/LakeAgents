#!/usr/bin/env python3
"""
"""
import json
import argparse
from pathlib import Path


def load_entries(log_path: str) -> list[dict]:
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def has_error(entry: dict) -> bool:
    return entry.get("selected_candidate_tables") == []


def main():
    parser = argparse.ArgumentParser(description="Filter experiment log entries with errors")
    parser.add_argument("log_file", type=str, help="Path to experiment log (JSONL)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file (optional)")
    parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    args = parser.parse_args()

    entries = load_entries(args.log_file)
    error_entries = [e for e in entries if has_error(e)]

    if args.format == "table":
        print(f"Found {len(error_entries)} error(s) in {len(entries)} total entries\n")
        print(f"{'session_id':<12} {'tau':<6} {'beta':<6} {'join_table':<20} error")
        print("-" * 100)
        for e in error_entries:
            sid = e.get("session_id", "?")
            tau = e.get("tau", "?")
            beta = e.get("beta", "?")
            tbl = e.get("join_table", "?")
            err = (e.get("error", "") or "")[:60]
            print(f"{sid:<12} {tau:<6} {beta:<6} {tbl:<20} {err}")
    else:
        print(json.dumps(error_entries, indent=2, ensure_ascii=False))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(error_entries, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()