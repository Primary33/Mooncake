#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path


KEY_FIELDS = (
    "random_input_len",
    "random_output_len",
    "request_rate",
    "max_concurrency",
)

METRICS = (
    "completed",
    "duration",
    "request_throughput",
    "input_throughput",
    "output_throughput",
    "total_throughput",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "p95_itl_ms",
    "p99_itl_ms",
    "mean_e2e_latency_ms",
    "p90_e2e_latency_ms",
    "p99_e2e_latency_ms",
    "concurrency",
    "max_output_tokens_per_s",
    "max_concurrent_requests",
)


def load_results(run_root: Path):
    rows = []
    for jsonl_path in sorted(run_root.glob("*/*/*.jsonl")):
        if jsonl_path.parent.name != "results":
            continue
        backend = jsonl_path.parent.parent.name
        with jsonl_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_backend_mode"] = backend
                row["_source_file"] = jsonl_path.name
                row["_line_no"] = line_no
                rows.append(row)
    return rows


def sort_key(row):
    def normalize(value):
        if value == "trace":
            return (2, str(value))
        if isinstance(value, (int, float)):
            return (0, value)
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, str(value))

    return (
        row.get("_backend_mode", ""),
        normalize(row.get("random_input_len")),
        normalize(row.get("random_output_len")),
        normalize(row.get("request_rate")),
        normalize(row.get("max_concurrency")),
        row.get("_source_file", ""),
    )


def write_all_results(rows, output_path: Path):
    fieldnames = [
        "_backend_mode",
        "_source_file",
        "tag",
        "backend",
        "dataset_name",
        "random_input_len",
        "random_output_len",
        "request_rate",
        "max_concurrency",
    ] + list(METRICS)

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=sort_key):
            writer.writerow({name: row.get(name) for name in fieldnames})


def build_comparison_rows(rows):
    grouped = {}
    for row in rows:
        key = tuple(row.get(name) for name in KEY_FIELDS)
        grouped.setdefault(key, {})[row["_backend_mode"]] = row

    comparison_rows = []
    for key in sorted(grouped):
        item = {name: value for name, value in zip(KEY_FIELDS, key)}
        classic = grouped[key].get("classic")
        tent = grouped[key].get("tent")

        for metric in METRICS:
            item[f"classic_{metric}"] = classic.get(metric) if classic else None
            item[f"tent_{metric}"] = tent.get(metric) if tent else None

        if classic and tent:
            for metric in METRICS:
                classic_value = classic.get(metric)
                tent_value = tent.get(metric)
                if isinstance(classic_value, (int, float)) and isinstance(
                    tent_value, (int, float)
                ):
                    delta = tent_value - classic_value
                    item[f"tent_minus_classic_{metric}"] = delta
                    if classic_value != 0:
                        item[f"tent_vs_classic_{metric}_pct"] = (
                            delta / classic_value
                        ) * 100.0
                    else:
                        item[f"tent_vs_classic_{metric}_pct"] = None
                else:
                    item[f"tent_minus_classic_{metric}"] = None
                    item[f"tent_vs_classic_{metric}_pct"] = None
        comparison_rows.append(item)

    return comparison_rows


def write_comparison(rows, output_path: Path):
    fieldnames = list(KEY_FIELDS)
    for metric in METRICS:
        fieldnames.append(f"classic_{metric}")
        fieldnames.append(f"tent_{metric}")
        fieldnames.append(f"tent_minus_classic_{metric}")
        fieldnames.append(f"tent_vs_classic_{metric}_pct")

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} <run_root>", file=sys.stderr)
        return 1

    run_root = Path(sys.argv[1]).resolve()
    if not run_root.exists():
        print(f"Run root does not exist: {run_root}", file=sys.stderr)
        return 1

    rows = load_results(run_root)
    if not rows:
        print(f"No JSONL benchmark results found under {run_root}", file=sys.stderr)
        return 1

    write_all_results(rows, run_root / "all_results.csv")
    comparison_rows = build_comparison_rows(rows)
    write_comparison(comparison_rows, run_root / "comparison.csv")
    print(f"Wrote {run_root / 'all_results.csv'}")
    print(f"Wrote {run_root / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
