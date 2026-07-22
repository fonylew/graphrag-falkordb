"""Turn the raw CSVs from run_batch_size_experiment.py into a short markdown
insights report: does per-file ingest time stay flat as batch size grows,
does finalize() cost grow faster than the graph, and which files/pages were
the slowest.
"""

import argparse
import csv
import os
import statistics
from collections import defaultdict


def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def to_int(v, default=None):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    std_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    std_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if std_x == 0 or std_y == 0:
        return None
    return cov / (std_x * std_y)


def models_used(rows: list[dict]) -> str:
    llms = sorted({r["llm_model"] for r in rows if r.get("llm_model")})
    embeds = sorted({r["embed_model"] for r in rows if r.get("embed_model")})
    if not llms and not embeds:
        return "_Model not recorded (older CSV format)._"
    parts = []
    if llms:
        parts.append(f"LLM: `{', '.join(llms)}`" + (" (mixed across rows!)" if len(llms) > 1 else ""))
    if embeds:
        parts.append(f"Embedder: `{', '.join(embeds)}`" + (" (mixed across rows!)" if len(embeds) > 1 else ""))
    return " | ".join(parts)


def summarize_isolated(rows: list[dict]) -> str:
    lines = [
        "## Isolated sweep (fresh empty graph per batch size)", "",
        models_used(rows), "",
        "| Batch Size | Trials | Avg Ingest (s) | Avg Finalize (s) | Avg Total (s) | Avg Sec/File | Failed Files |",
        "|---|---|---|---|---|---|---|",
    ]
    by_size = defaultdict(list)
    for r in rows:
        by_size[to_int(r["batch_size"])].append(r)

    for size in sorted(by_size):
        group = by_size[size]
        avg_ingest = statistics.mean(to_float(r["ingest_duration_sec"]) for r in group)
        avg_final = statistics.mean(to_float(r["finalize_duration_sec"]) for r in group)
        avg_total = statistics.mean(to_float(r["total_duration_sec"]) for r in group)
        avg_per_file = statistics.mean(to_float(r["avg_sec_per_file"]) for r in group)
        failed = sum(to_int(r["failed_files"], 0) or 0 for r in group)
        lines.append(f"| {size} | {len(group)} | {avg_ingest:.2f} | {avg_final:.2f} | "
                      f"{avg_total:.2f} | {avg_per_file:.2f} | {failed} |")

    per_file_avgs = [statistics.mean(to_float(r["avg_sec_per_file"]) for r in by_size[s]) for s in sorted(by_size)]
    if len(per_file_avgs) >= 2:
        mean_v = statistics.mean(per_file_avgs)
        cv = (statistics.pstdev(per_file_avgs) / mean_v) if mean_v else 0
        verdict = ("Roughly constant per-file time (supports linear ingest-time scaling)." if cv < 0.20
                   else "Notable drift in per-file time as batch size grows — worth checking whether this is "
                        "LLM warm-up, thermal throttling, or a graph-size effect leaking into the isolated design.")
        lines += ["", f"Per-file ingest time coefficient of variation across batch sizes: **{cv:.1%}**. {verdict}"]

    return "\n".join(lines)


def summarize_cumulative(rows: list[dict]) -> str:
    lines = [
        "## Cumulative sweep (single growing graph)", "",
        models_used(rows), "",
        "| Cumulative Files | Trials | Avg Ingest (s) | Avg Finalize (s) | Avg Total (s) | Avg Entities | Failed Files |",
        "|---|---|---|---|---|---|---|",
    ]
    by_size = defaultdict(list)
    for r in rows:
        by_size[to_int(r["cumulative_files"])].append(r)

    for size in sorted(by_size):
        group = by_size[size]
        avg_ingest = statistics.mean(to_float(r["ingest_duration_sec"]) for r in group)
        avg_final = statistics.mean(to_float(r["finalize_duration_sec"]) for r in group)
        avg_total = statistics.mean(to_float(r["total_duration_sec"]) for r in group)
        avg_entities = statistics.mean(to_float(r["entity_count"]) for r in group)
        failed = sum(to_int(r["failed_files"], 0) or 0 for r in group)
        lines.append(f"| {size} | {len(group)} | {avg_ingest:.2f} | {avg_final:.2f} | "
                      f"{avg_total:.2f} | {avg_entities:.0f} | {failed} |")

    sizes_sorted = sorted(by_size)
    if len(sizes_sorted) >= 2:
        first_final = statistics.mean(to_float(r["finalize_duration_sec"]) for r in by_size[sizes_sorted[0]])
        last_final = statistics.mean(to_float(r["finalize_duration_sec"]) for r in by_size[sizes_sorted[-1]])
        file_growth = sizes_sorted[-1] / sizes_sorted[0] if sizes_sorted[0] else float("inf")
        final_growth = (last_final / first_final) if first_final else float("inf")
        verdict = ("This growth outpaces file-count growth, supporting Hypothesis 2 from "
                    "ingestion_experiment_test_plan.md (finalize scales with total graph size, not batch size)."
                   if final_growth > file_growth else
                    "Growth roughly tracks file-count growth here — finalize scaling looks closer to linear "
                    "at this graph size; the effect may still show up at larger scale.")
        lines += ["", f"finalize() duration grew **{final_growth:.1f}x** from the smallest checkpoint "
                       f"({sizes_sorted[0]} files, {first_final:.2f}s) to the largest "
                       f"({sizes_sorted[-1]} files, {last_final:.2f}s), while file count grew "
                       f"{file_growth:.1f}x. {verdict}"]

    return "\n".join(lines)


def summarize_files(rows: list[dict], top_n: int = 10) -> str:
    lines = [
        "## Slowest individual files", "",
        models_used(rows), "",
        "| Ingest (s) | Pages | LLM Model | File Path |", "|---|---|---|---|",
    ]
    ranked = sorted(rows, key=lambda r: to_float(r["ingest_duration_sec"]), reverse=True)[:top_n]
    for r in ranked:
        pages = r.get("page_count") or "?"
        model = r.get("llm_model") or "?"
        lines.append(f"| {to_float(r['ingest_duration_sec']):.2f} | {pages} | {model} | {r['file_path']} |")

    pairs = [
        (to_int(r.get("page_count")), to_float(r["ingest_duration_sec"]))
        for r in rows
        if to_int(r.get("page_count")) is not None and r.get("status") == "SUCCESS"
    ]
    if len(pairs) >= 3:
        xs, ys = zip(*pairs)
        r_val = pearson(list(xs), list(ys))
        if r_val is not None:
            verdict = ("Longer PDFs meaningfully take longer to ingest." if r_val > 0.4
                       else "Page count alone is a weak predictor of ingest time — chunk count and content "
                            "density likely matter more than raw page count.")
            lines += ["", f"Pearson correlation between page count and ingest duration: **{r_val:.2f}** "
                           f"(n={len(pairs)}). {verdict}"]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize batch-size ingestion experiment results")
    parser.add_argument("--isolated-log", default="results/batch_size_isolated.csv")
    parser.add_argument("--cumulative-log", default="results/batch_size_cumulative.csv")
    parser.add_argument("--isolated-files-log", default="results/batch_size_isolated_files.csv")
    parser.add_argument("--cumulative-files-log", default="results/batch_size_cumulative_files.csv")
    parser.add_argument("--out", default="results/batch_size_insights.md")
    args = parser.parse_args()

    sections = ["# Batch-Size Ingestion Experiment: Insights", ""]

    isolated_rows = read_csv(args.isolated_log)
    sections.append(summarize_isolated(isolated_rows) if isolated_rows
                     else f"_No isolated results found at {args.isolated_log}._")
    sections.append("")

    cumulative_rows = read_csv(args.cumulative_log)
    sections.append(summarize_cumulative(cumulative_rows) if cumulative_rows
                     else f"_No cumulative results found at {args.cumulative_log}._")
    sections.append("")

    file_rows = read_csv(args.isolated_files_log) + read_csv(args.cumulative_files_log)
    sections.append(summarize_files(file_rows) if file_rows else "_No per-file results found._")
    sections.append("")

    report = "\n".join(sections)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)

    print(report)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
