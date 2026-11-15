from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


INT_FIELDS = {
    "target": re.compile(br'"attention_target_token_index"\s*:\s*(\d+)'),
    "budget": re.compile(br'"budget_bucket_k"\s*:\s*(\d+)'),
    "required_k": re.compile(br'"required_k"\s*:\s*(\d+)'),
}
COVERAGE = re.compile(br'"oracle_coverage_threshold"\s*:\s*([0-9.eE+-]+)')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Oracle budget labels by effective causal context length."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument(
        "--length-edges",
        nargs="+",
        type=int,
        default=[4096, 8192, 16384, 24576, 32768],
        help="Inclusive upper bounds for context-length bins.",
    )
    parser.add_argument(
        "--budget-buckets", nargs="+", type=int, default=[0, 4, 8, 16, 32]
    )
    parser.add_argument(
        "--budget-bucket-max",
        type=int,
        default=None,
        help="Use every integer budget bucket from 0 through this maximum.",
    )
    parser.add_argument("--expected-coverage", type=float, default=None)
    parser.add_argument(
        "--target-selector",
        default=None,
        help="Read required_k/budget_bucket_k from budget_targets[selector].",
    )
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def extract_int(pattern: re.Pattern[bytes], line: bytes) -> int:
    match = pattern.search(line)
    if match is None:
        raise ValueError
    return int(match.group(1))


def extract_target_fields(line: bytes, selector: str | None) -> tuple[int, int]:
    if selector is None:
        return (
            extract_int(INT_FIELDS["budget"], line),
            extract_int(INT_FIELDS["required_k"], line),
        )
    selector_pattern = re.compile(
        rb'"' + re.escape(selector.encode("utf-8")) + rb'"\s*:\s*\{([^{}]*)\}'
    )
    match = selector_pattern.search(line)
    if match is None:
        raise ValueError
    payload = match.group(1)
    return (
        extract_int(INT_FIELDS["budget"], payload),
        extract_int(INT_FIELDS["required_k"], payload),
    )


def bin_name(length: int, edges: list[int]) -> str:
    lower = 0
    for upper in edges:
        if length <= upper:
            return f"{lower // 1024}-{upper // 1024}K"
        lower = upper
    return f">{edges[-1] // 1024}K"


def main() -> None:
    args = parse_args()
    edges = sorted(set(args.length_edges))
    if args.budget_bucket_max is not None:
        if args.budget_bucket_max < 0:
            raise ValueError("--budget-bucket-max must be non-negative.")
        args.budget_buckets = list(range(args.budget_bucket_max + 1))
    buckets = sorted(set(args.budget_buckets))
    if not edges or edges[0] <= 0:
        raise ValueError("--length-edges must contain positive values.")
    if not buckets or buckets[0] < 0:
        raise ValueError("--budget-buckets must contain non-negative values.")

    counts: dict[str, Counter[int]] = defaultdict(Counter)
    required_sum: Counter[str] = Counter()
    required_over_max: Counter[str] = Counter()
    total = 0
    malformed = 0
    observed_coverage: float | None = None

    with Path(args.input_jsonl).open("rb", buffering=16 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                target = extract_int(INT_FIELDS["target"], line)
                budget, required_k = extract_target_fields(line, args.target_selector)
            except ValueError:
                malformed += 1
                continue

            match = COVERAGE.search(line)
            if match is None:
                raise ValueError(
                    f"Missing oracle_coverage_threshold at line {line_number}."
                )
            row_coverage = float(match.group(1))
            if observed_coverage is None:
                observed_coverage = row_coverage
            elif abs(row_coverage - observed_coverage) > 1e-9:
                raise ValueError(
                    f"Mixed coverage thresholds at line {line_number}: "
                    f"first={observed_coverage}, current={row_coverage}."
                )
            if budget not in buckets:
                raise ValueError(
                    f"Unknown budget bucket {budget} at line {line_number}; "
                    f"expected one of {buckets}."
                )

            name = bin_name(target + 1, edges)
            counts[name][budget] += 1
            required_sum[name] += required_k
            required_over_max[name] += int(required_k > buckets[-1])
            total += 1

    if total == 0:
        raise ValueError(f"No valid Oracle rows found in {args.input_jsonl}.")
    if malformed:
        raise ValueError(
            f"Found {malformed} malformed Oracle rows in {args.input_jsonl}."
        )
    if (
        args.expected_coverage is not None
        and observed_coverage is not None
        and abs(observed_coverage - args.expected_coverage) > 1e-9
    ):
        raise ValueError(
            f"Coverage mismatch: file={observed_coverage}, "
            f"expected={args.expected_coverage}."
        )

    names = [bin_name(edge, edges) for edge in edges]
    overflow = f">{edges[-1] // 1024}K"
    if overflow in counts:
        names.append(overflow)

    records: list[dict[str, object]] = []
    header = ["length", "rows", "data_share"] + [f"budget_{b}" for b in buckets]
    header += ["avg_required_k", f"required_k_gt_{buckets[-1]}"]
    print("\t".join(header))
    for name in names:
        rows = sum(counts[name].values())
        if rows == 0:
            continue
        record: dict[str, object] = {
            "length_bin": name,
            "rows": rows,
            "data_share": rows / total,
            "budget_distribution": {
                str(bucket): counts[name][bucket] / rows for bucket in buckets
            },
            "avg_required_k": required_sum[name] / rows,
            "required_k_over_max_rate": required_over_max[name] / rows,
            "actual_budget_distribution": {
                **{
                    str(bucket): (
                        counts[name][bucket]
                        - (required_over_max[name] if bucket == buckets[-1] else 0)
                    )
                    / rows
                    for bucket in buckets
                },
                f">{buckets[-1]}": required_over_max[name] / rows,
            },
        }
        records.append(record)
        values = [name, str(rows), f"{100 * rows / total:.2f}%"]
        values.extend(
            f"{100 * (counts[name][bucket] - (required_over_max[name] if bucket == buckets[-1] else 0)) / rows:.2f}%"
            for bucket in buckets
        )
        values.extend(
            [
                f"{required_sum[name] / rows:.2f}",
                f"{100 * required_over_max[name] / rows:.2f}%",
            ]
        )
        print("\t".join(values))

    overall_counts = {bucket: sum(row[bucket] for row in counts.values()) for bucket in buckets}
    overall_required_over_max = sum(required_over_max.values())
    print(
        "overall\t"
        + str(total)
        + "\t100.00%\t"
        + "\t".join(
            f"{100 * (overall_counts[b] - (overall_required_over_max if b == buckets[-1] else 0)) / total:.2f}%"
            for b in buckets
        )
    )

    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "input_jsonl": str(args.input_jsonl),
                    "target_selector": args.target_selector,
                    "coverage": observed_coverage,
                    "total_rows": total,
                    "malformed_rows": malformed,
                    "budget_buckets": buckets,
                    "length_edges": edges,
                    "bins": records,
                    "overall_budget_distribution": {
                        str(bucket): overall_counts[bucket] / total for bucket in buckets
                    },
                    "overall_actual_budget_distribution": {
                        **{
                            str(bucket): (
                                overall_counts[bucket]
                                - (overall_required_over_max if bucket == buckets[-1] else 0)
                            )
                            / total
                            for bucket in buckets
                        },
                        f">{buckets[-1]}": overall_required_over_max / total,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
