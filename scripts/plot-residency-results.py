#!/usr/bin/env python3
"""Summarize residency-bench.py JSONL results and generate PNG charts."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(paths):
    rows = []

    for path in paths:
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(f"WARNING: Skipping malformed JSON in {path}")
                    continue

                row["source_file"] = path.name
                row["model_tag"] = path.name.split("_cold_")[0].split("_warm_")[0]
                rows.append(row)

    return rows


def successful(row):
    return row.get("rc") == 0


def number(row, key):
    value = row.get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def residency_value(row, key):
    after = row.get("resident_after", {})
    return after.get(key)


def median_or_blank(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="Benchmark JSONL files")
    parser.add_argument("--out-dir", default="results/plots")
    args = parser.parse_args()

    paths = [Path(p) for p in args.files]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        parser.error("Files not found: " + ", ".join(missing))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_results(paths)
    if not rows:
        parser.error("No benchmark records found.")

    ok = [r for r in rows if successful(r)]
    failed = [r for r in rows if not successful(r)]

    print(f"Loaded {len(rows)} runs: {len(ok)} successful, {len(failed)} failed.")
    if failed:
        print("Failed runs:")
        for r in failed:
            print(f"  {r['source_file']}: mode={r.get('mode')} rc={r.get('rc')}")

    # Save a compact summary. Failed runs remain visible through run counts.
    summary_path = out_dir / "summary.csv"
    groups = sorted({
        (r.get("model_tag", "model"), r.get("state", "unknown"), r.get("mode", "unknown"))
        for r in rows
    })

    fields = [
        "model", "state", "mode", "runs", "successful", "failed",
        "median_ttft_ms", "median_load_ms", "median_wall_s",
        "median_eval_tps", "median_major_faults", "median_max_rss_mib",
        "median_hot_resident_mib", "median_cold_resident_mib",
        "median_cold_total_mib",
    ]

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for model, state, mode in groups:
            subset = [
                r for r in rows
                if r.get("model_tag", "model") == model
                and r.get("state", "unknown") == state
                and r.get("mode", "unknown") == mode
            ]
            good = [r for r in subset if successful(r)]

            def vals(key):
                return [number(r, key) for r in good]

            rss = [number(r, "max_rss_kb") for r in good]
            hot = [residency_value(r, "hot_resident_mib") for r in good]
            cold = [residency_value(r, "cold_resident_mib") for r in good]
            cold_total = [residency_value(r, "cold_mib") for r in good]

            writer.writerow({
                "model": model,
                "state": state,
                "mode": mode,
                "runs": len(subset),
                "successful": len(good),
                "failed": len(subset) - len(good),
                "median_ttft_ms": median_or_blank(vals("ttft_ms")),
                "median_load_ms": median_or_blank(vals("load_ms")),
                "median_wall_s": median_or_blank(vals("wall_s")),
                "median_eval_tps": median_or_blank(vals("eval_tps")),
                "median_major_faults": median_or_blank(vals("major_faults")),
                "median_max_rss_mib": median_or_blank(
                    [v / 1024 for v in rss if v is not None]
                ),
                "median_hot_resident_mib": median_or_blank(hot),
                "median_cold_resident_mib": median_or_blank(cold),
                "median_cold_total_mib": median_or_blank(cold_total),
            })

    # Compare individual successful runs, not just medians, so variability is visible.
    metrics = [
        ("ttft_ms", "Time to first token (ms)", "ttft"),
        ("load_ms", "Load time (ms)", "load_time"),
        ("eval_tps", "Decode throughput (tokens/s)", "decode_tps"),
        ("major_faults", "Major page faults per run", "major_faults"),
        ("max_rss_kb", "Maximum RSS (MiB)", "max_rss"),
    ]

    for field, ylabel, filename in metrics:
        labels = []
        data = []

        for model, state, mode in groups:
            values = []
            for r in ok:
                if (
                    r.get("model_tag", "model") == model
                    and r.get("state", "unknown") == state
                    and r.get("mode", "unknown") == mode
                ):
                    value = number(r, field)
                    if value is not None:
                        values.append(value / 1024 if field == "max_rss_kb" else value)

            if values:
                labels.append(f"{state}\n{mode}")
                data.append(values)

        if not data:
            continue

        fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.25), 5))
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " across successful runs")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out_path = out_dir / f"{filename}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"Wrote {out_path}")

    # Cold expert residency, when the model/harness reports any cold pages.
    residency_groups = []
    for model, state, mode in groups:
        values = []
        for r in ok:
            if (
                r.get("model_tag", "model") == model
                and r.get("state", "unknown") == state
                and r.get("mode", "unknown") == mode
            ):
                total = residency_value(r, "cold_mib")
                resident = residency_value(r, "cold_resident_mib")
                if total is not None and total > 0 and resident is not None:
                    values.append(100 * resident / total)
        if values:
            residency_groups.append((f"{state}\n{mode}", values))

    if residency_groups:
        fig, ax = plt.subplots(
            figsize=(max(7, len(residency_groups) * 1.25), 5)
        )
        ax.boxplot(
            [v for _, v in residency_groups],
            tick_labels=[label for label, _ in residency_groups],
            showmeans=True,
        )
        ax.set_ylabel("Cold expert pages resident (%)")
        ax.set_title("Cold expert page-cache residency")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out_path = out_dir / "cold_residency_percent.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"Wrote {out_path}")
    else:
        print(
            "No cold-residency chart: the supplied results have no nonzero "
            "cold tensor total. Verify this is an MoE model and tensor "
            "classification matches the C++ implementation."
        )

    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()