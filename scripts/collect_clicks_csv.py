#!/usr/bin/env python3
"""
collect_clicks_csv.py — Collect confirmed clicks from labeled CSVs
==================================================================

Scans selected directories (non-recursive) for CSV files produced by
DataCollectionDialogV5, and merges ONLY the rows where label == 1
(confirmed clicks) into a single output CSV.

No pipeline re-run. No .paudio files. Pure CSV filtering.

Usage (run from the project root):
    python scripts/collect_clicks_csv.py <dir1> [<dir2> ...] [options]

Examples:
    python scripts/collect_clicks_csv.py ~/data/aloe ~/data/cactus
    python scripts/collect_clicks_csv.py ~/data/aloe --output clicks_dataset.csv

Options:
    --output PATH   Output CSV path (default: confirmed_clicks.csv)

Notes:
    - A 'session_id' column is added automatically (= file column value,
      used as group key for StratifiedGroupKFold in train_svm.py).
    - Rows with label != 1 are silently skipped.
    - All source CSV columns are preserved in the output.
"""

import sys
import csv
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Collect label=1 rows from DataCollectionDialogV5 CSVs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'directories', nargs='+', type=Path,
        help='Directories to scan for CSV files (non-recursive)',
    )
    parser.add_argument(
        '--output', type=Path, default=Path('confirmed_clicks.csv'),
        help='Output CSV path (default: confirmed_clicks.csv)',
    )
    args = parser.parse_args()

    # Collect all CSV files from selected directories (non-recursive)
    all_csv_files: list[Path] = []
    for d in args.directories:
        if not d.is_dir():
            print(f"Warning: {d} is not a directory — skipping")
            continue
        found = sorted(d.glob('*.csv'))
        print(f"{d.name}/  →  {len(found)} CSV file(s)")
        all_csv_files.extend(found)

    if not all_csv_files:
        print("No CSV files found in the specified directories.")
        sys.exit(1)

    print(f"\nTotal: {len(all_csv_files)} CSV file(s)\n")

    all_rows:    list[dict] = []
    fieldnames:  list[str]  = []
    files_hit:   int        = 0

    for csv_path in all_csv_files:
        try:
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                if reader.fieldnames is None:
                    print(f"  {csv_path.name}: empty or unreadable — skipping")
                    continue

                # Capture column order from first file that has data
                if not fieldnames:
                    fieldnames = list(reader.fieldnames)
                    # Prepend session_id if not already present
                    if 'session_id' not in fieldnames:
                        fieldnames = ['session_id'] + fieldnames

                # Non-numeric columns: never touch their content
                _skip_norm = {'session_id', 'file', 'label', '', None}

                rows_from_file = 0
                for row in reader:
                    try:
                        label_val = str(row.get('label', '')).strip()
                        if label_val != '1':
                            continue

                        # session_id = FULL file value (NOT .stem).
                        # .stem truncates filenames that contain dots — e.g.
                        # "..._13.49" → "..._13" — which can split one recording
                        # into two groups or merge two recordings into one,
                        # corrupting StratifiedGroupKFold. The full file string
                        # is the only reliable per-recording key.
                        file_col = row.get('file') or csv_path.stem
                        row['session_id'] = file_col

                        # Normalize European decimal commas → dots in numeric
                        # columns ("12,73" → "12.73"). Excel on an Italian
                        # locale silently rewrites numbers this way, and pandas
                        # would then read them as strings → astype(float) crashes.
                        for col, val in list(row.items()):
                            if col in _skip_norm or not isinstance(val, str):
                                continue
                            v = val.strip()
                            if ',' in v and '.' not in v:
                                row[col] = v.replace(',', '.')

                        all_rows.append(row)
                        rows_from_file += 1
                    except Exception:
                        continue

                if rows_from_file > 0:
                    files_hit += 1
                print(f"  {csv_path.name}: {rows_from_file} confirmed click(s)")

        except Exception as e:
            print(f"  {csv_path.name}: error — {e}")
            continue

    if not all_rows:
        print("\nNo label=1 rows found across all files.")
        sys.exit(0)

    # Write merged output
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Ensure every row dict has all fieldnames (fill missing with empty string)
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})

    print(f"\n{len(all_rows)} confirmed clicks from {files_hit} file(s)")
    print(f"Saved to: {args.output}")
    print(f"\nNext: python scripts/train_svm.py --csv {args.output}")


if __name__ == '__main__':
    main()
