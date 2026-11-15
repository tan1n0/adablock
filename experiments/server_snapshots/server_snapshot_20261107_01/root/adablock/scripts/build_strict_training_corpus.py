from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


SOURCE_FILES = {
    "pg19": (
        "emozilla/pg19",
        "data/train-00000-of-00023-5263dc4323e881d2.parquet",
        "parquet",
    ),
    "booksum": ("kmfoda/booksum", "train.csv", "csv"),
    "arxiv": (
        "ccdv/arxiv-summarization",
        "document/train-00000-of-00015.parquet",
        "parquet",
    ),
    "pubmed": (
        "ccdv/pubmed-summarization",
        "document/train-00000-of-00005.parquet",
        "parquet",
    ),
    "wikihop": (
        "MoE-UNC/wikihop",
        "data/train-00000-of-00001.parquet",
        "parquet",
    ),
}

DEFAULT_COUNTS = {
    "pg19": 750,
    "booksum": 750,
    "arxiv": 1000,
    "pubmed": 1000,
    "wikihop": 1000,
    "multi_needle": 1500,
}

LENGTH_BUCKETS = (6000, 12000, 20000, 32000)
LENGTH_WEIGHTS = (0.25, 0.30, 0.30, 0.15)

FILLER_SENTENCES = (
    "This paragraph contains background material that is unrelated to the requested evidence.",
    "Several entities are discussed here to create realistic retrieval distractors.",
    "The report continues with ordinary operational details and contextual observations.",
    "Readers must distinguish repeated terminology from the exact fact requested later.",
    "This section increases context length without changing the final answer.",
    "Additional narrative information appears here as a hard negative for retrieval.",
    "The document records routine events, dates, and names that are not part of the answer.",
    "A superficially similar statement is included, but it refers to a different entity.",
)

CITIES = (
    "Lisbon",
    "Nairobi",
    "Reykjavik",
    "Kyoto",
    "Valencia",
    "Tallinn",
    "Montreal",
    "Adelaide",
)

AGENTS = (
    "Mara Venn",
    "Jonas Pike",
    "Leila North",
    "Tomas Reed",
    "Anika Sol",
    "Ravi Hale",
    "Nora Quinn",
    "Elias Ward",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a strict non-LongBench AdaBlock training corpus."
    )
    parser.add_argument("--train-output", default="data/train_strict_clean.jsonl")
    parser.add_argument(
        "--validation-output", default="data/validation_strict_clean.jsonl"
    )
    parser.add_argument(
        "--metadata-output", default="data/train_strict_clean_metadata.json"
    )
    parser.add_argument("--cache-dir", default=".cache/adablock_training_sources")
    parser.add_argument("--longbench-dir", default="data/longbench")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=1500)
    parser.add_argument("--pg19-count", type=int, default=DEFAULT_COUNTS["pg19"])
    parser.add_argument("--booksum-count", type=int, default=DEFAULT_COUNTS["booksum"])
    parser.add_argument("--arxiv-count", type=int, default=DEFAULT_COUNTS["arxiv"])
    parser.add_argument("--pubmed-count", type=int, default=DEFAULT_COUNTS["pubmed"])
    parser.add_argument("--wikihop-count", type=int, default=DEFAULT_COUNTS["wikihop"])
    parser.add_argument(
        "--multi-needle-count", type=int, default=DEFAULT_COUNTS["multi_needle"]
    )
    parser.add_argument("--decontamination-ngram-words", type=int, default=13)
    parser.add_argument("--decontamination-stride", type=int, default=5)
    parser.add_argument("--decontamination-min-matches", type=int, default=3)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(clean_text(item) for item in value if item is not None)
    if isinstance(value, dict):
        return "\n".join(clean_text(item) for item in value.values() if item is not None)
    return " ".join(str(value).split())


def stable_hash(value: str, length: int = 20) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def ngram_hash(words: list[str]) -> int:
    digest = hashlib.blake2b(" ".join(words).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def sampled_ngram_hashes(
    text: str,
    ngram_words: int,
    stride: int,
) -> Iterator[int]:
    words = normalized_words(text)
    if len(words) < ngram_words:
        return
    for start in range(0, len(words) - ngram_words + 1, max(stride, 1)):
        yield ngram_hash(words[start : start + ngram_words])


class LongBenchDecontaminator:
    def __init__(
        self,
        data_dir: Path,
        ngram_words: int,
        stride: int,
        min_matches: int,
    ) -> None:
        self.ngram_words = ngram_words
        self.stride = stride
        self.min_matches = min_matches
        self.hashes: set[int] = set()
        self.files: list[str] = []
        for path in sorted(data_dir.glob("*.jsonl")):
            self.files.append(str(path))
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    context = clean_text(row.get("context", ""))
                    self.hashes.update(
                        sampled_ngram_hashes(
                            context,
                            ngram_words=self.ngram_words,
                            stride=self.stride,
                        )
                    )

    def match_count(self, text: str) -> int:
        matches = 0
        for fingerprint in sampled_ngram_hashes(
            text,
            ngram_words=self.ngram_words,
            stride=self.stride,
        ):
            if fingerprint in self.hashes:
                matches += 1
                if matches >= self.min_matches:
                    break
        return matches

    def contaminated(self, text: str) -> bool:
        return self.match_count(text) >= self.min_matches


def choose_length(rng: random.Random) -> int:
    return rng.choices(LENGTH_BUCKETS, weights=LENGTH_WEIGHTS, k=1)[0]


def crop_context(text: str, max_chars: int, rng: random.Random) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""
    mode = rng.choice(("start_end", "window"))
    if mode == "window":
        start = rng.randint(0, max(len(text) - max_chars, 0))
        return text[start : start + max_chars].strip()
    first = max_chars // 2
    return (text[:first] + "\n...[middle omitted]...\n" + text[-(max_chars - first) :]).strip()


def fit_prompt(
    prefix: str,
    context: str,
    suffix: str,
    max_chars: int,
    rng: random.Random,
) -> str:
    reserved = len(prefix) + len(suffix) + 4
    context_budget = max(0, max_chars - reserved)
    cropped = crop_context(context, context_budget, rng)
    return f"{prefix}{cropped}{suffix}".strip()


def iter_parquet_rows(path: Path) -> Iterator[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=8):
        yield from batch.to_pylist()


def iter_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    csv.field_size_limit(2**31 - 1)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        yield from csv.DictReader(handle)


def download_source(source: str, cache_dir: Path) -> Path:
    repo_id, filename, _kind = SOURCE_FILES[source]
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        cache_dir=str(cache_dir),
    )
    return Path(path)


def sentence_candidates(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", clean_text(text))
    return [
        sentence
        for sentence in sentences
        if 60 <= len(sentence) <= 360 and len(sentence.split()) >= 10
    ]


def format_pg19(row: dict[str, Any], max_chars: int, rng: random.Random) -> str | None:
    raw_text = clean_text(row.get("text"))
    if not raw_text:
        return None
    context_budget = max(max_chars - 700, 1000)
    context = crop_context(raw_text, context_budget, rng)
    candidates = sentence_candidates(context)
    if not candidates:
        return None
    answer = rng.choice(candidates)
    cue = " ".join(answer.split()[:8])
    title = clean_text(row.get("short_book_title")) or "an untitled book"
    suffix = (
        f"\n\nQuestion:\nIn {title}, reproduce the sentence that begins with "
        f'\"{cue}\".\n\nAnswer:\n{answer}'
    )
    return fit_prompt("Context:\n", context, suffix, max_chars, rng)


def first_nonempty(row: dict[str, Any], names: Iterable[str]) -> str:
    for name in names:
        value = clean_text(row.get(name))
        if value:
            return value
    return ""


def format_booksum(row: dict[str, Any], max_chars: int, rng: random.Random) -> str | None:
    chapter = first_nonempty(row, ("chapter", "content", "text", "document"))
    summary = first_nonempty(row, ("summary_text", "summary", "target"))
    if not chapter or not summary:
        return None
    summary = summary[:4000]
    suffix = f"\n\nTask:\nSummarize the important events in this chapter.\n\nSummary:\n{summary}"
    return fit_prompt("Chapter:\n", chapter, suffix, max_chars, rng)


def format_scientific(
    row: dict[str, Any],
    max_chars: int,
    rng: random.Random,
    source: str,
) -> str | None:
    article = clean_text(row.get("article"))
    abstract = clean_text(row.get("abstract"))
    if not article or not abstract:
        return None
    abstract = abstract[:4000]
    suffix = (
        f"\n\nTask:\nWrite a concise evidence-based summary of this {source} paper."
        f"\n\nSummary:\n{abstract}"
    )
    return fit_prompt("Paper:\n", article, suffix, max_chars, rng)


def format_wikihop(row: dict[str, Any], max_chars: int, rng: random.Random) -> str | None:
    supports = row.get("supports") or []
    question = clean_text(row.get("query") or row.get("question")).replace("_", " ")
    answer = clean_text(row.get("answer"))
    if not supports or not question or not answer:
        return None
    indexed = [f"Document {idx + 1}: {clean_text(text)}" for idx, text in enumerate(supports)]
    context = "\n\n".join(indexed)
    suffix = f"\n\nQuestion:\n{question}\n\nAnswer:\n{answer}"
    return fit_prompt("Supporting documents:\n", context, suffix, max_chars, rng)


def make_multi_needle(index: int, max_chars: int, rng: random.Random) -> str:
    project = f"PROJECT-{index:05d}-{rng.randint(100, 999)}"
    agent = rng.choice(AGENTS)
    city = rng.choice(CITIES)
    decoy_agent = rng.choice([name for name in AGENTS if name != agent])
    decoy_city = rng.choice([name for name in CITIES if name != city])
    facts = [
        f"The project {project} is assigned to {agent}.",
        f"{agent} works from the regional office in {city}.",
        f"{decoy_agent} works from the regional office in {decoy_city}.",
    ]
    rng.shuffle(facts)

    paragraphs: list[str] = []
    target_body_chars = max(max_chars - 500, 1000)
    while len("\n".join(paragraphs)) < target_body_chars:
        paragraphs.append(rng.choice(FILLER_SENTENCES))
    for fact in facts:
        position = rng.randint(2, max(2, len(paragraphs) - 2))
        paragraphs.insert(position, fact)
    context = "\n".join(paragraphs)
    suffix = (
        f"\n\nQuestion:\nIn which city does the person assigned to {project} work?"
        f"\n\nAnswer:\n{city}"
    )
    return fit_prompt("Context:\n", context, suffix, max_chars, rng)


def source_rows(source: str, path: Path) -> Iterator[dict[str, Any]]:
    kind = SOURCE_FILES[source][2]
    if kind == "parquet":
        yield from iter_parquet_rows(path)
    else:
        yield from iter_csv_rows(path)


def collect_source(
    source: str,
    target_count: int,
    path: Path,
    rng: random.Random,
    min_chars: int,
    decontaminator: LongBenchDecontaminator,
    seen_ids: set[str],
    removed: Counter[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in source_rows(source, path):
        max_chars = choose_length(rng)
        if source == "pg19":
            text = format_pg19(raw, max_chars, rng)
        elif source == "booksum":
            text = format_booksum(raw, max_chars, rng)
        elif source in {"arxiv", "pubmed"}:
            text = format_scientific(raw, max_chars, rng, source)
        elif source == "wikihop":
            text = format_wikihop(raw, max_chars, rng)
        else:
            raise ValueError(source)
        if not text or len(text) < min_chars:
            removed[f"{source}:too_short_or_invalid"] += 1
            continue
        doc_id = stable_hash(text)
        if doc_id in seen_ids:
            removed[f"{source}:duplicate"] += 1
            continue
        if decontaminator.contaminated(text):
            removed[f"{source}:longbench_overlap"] += 1
            continue
        seen_ids.add(doc_id)
        rows.append({"doc_id": doc_id, "source": source, "text": text})
        if len(rows) >= target_count:
            break
    if len(rows) < target_count:
        raise RuntimeError(
            f"Source {source} produced {len(rows)} rows, below target {target_count}."
        )
    return rows


def collect_multi_needle(
    target_count: int,
    rng: random.Random,
    min_chars: int,
    decontaminator: LongBenchDecontaminator,
    seen_ids: set[str],
    removed: Counter[str],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    index = 0
    while len(rows) < target_count:
        text = make_multi_needle(index, choose_length(rng), rng)
        index += 1
        if len(text) < min_chars:
            removed["multi_needle:too_short"] += 1
            continue
        doc_id = stable_hash(text)
        if doc_id in seen_ids:
            removed["multi_needle:duplicate"] += 1
            continue
        if decontaminator.contaminated(text):
            removed["multi_needle:longbench_overlap"] += 1
            continue
        seen_ids.add(doc_id)
        rows.append({"doc_id": doc_id, "source": "multi_needle", "text": text})
    return rows


def stratified_split(
    rows: list[dict[str, str]],
    validation_ratio: float,
    rng: random.Random,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_source: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    train: list[dict[str, str]] = []
    validation: list[dict[str, str]] = []
    for source in sorted(by_source):
        source_rows_list = by_source[source]
        rng.shuffle(source_rows_list)
        val_size = max(1, int(round(len(source_rows_list) * validation_ratio)))
        validation.extend(source_rows_list[:val_size])
        train.extend(source_rows_list[val_size:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def write_jsonl(path: Path, rows: list[dict[str, str]], split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({**row, "split": split}, ensure_ascii=False) + "\n")


def count_sources(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row["source"] for row in rows).items()))


def length_bucket_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        length = len(row["text"])
        if length <= 6000:
            bucket = "<=6k"
        elif length <= 12000:
            bucket = "6k-12k"
        elif length <= 20000:
            bucket = "12k-20k"
        else:
            bucket = "20k-32k"
        counts[bucket] += 1
    return dict(counts)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be in (0, 1).")
    rng = random.Random(args.seed)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    decontaminator = LongBenchDecontaminator(
        data_dir=Path(args.longbench_dir),
        ngram_words=args.decontamination_ngram_words,
        stride=args.decontamination_stride,
        min_matches=args.decontamination_min_matches,
    )
    if not decontaminator.files:
        raise FileNotFoundError(
            f"No LongBench JSONL files found in {args.longbench_dir}; "
            "strict decontamination cannot be verified."
        )

    targets = {
        "pg19": args.pg19_count,
        "booksum": args.booksum_count,
        "arxiv": args.arxiv_count,
        "pubmed": args.pubmed_count,
        "wikihop": args.wikihop_count,
    }
    downloaded: dict[str, str] = {}
    seen_ids: set[str] = set()
    removed: Counter[str] = Counter()
    all_rows: list[dict[str, str]] = []

    for source, target in targets.items():
        if target <= 0:
            continue
        print({"event": "download_source", "source": source, "target": target}, flush=True)
        path = download_source(source, cache_dir)
        downloaded[source] = str(path)
        rows = collect_source(
            source=source,
            target_count=target,
            path=path,
            rng=rng,
            min_chars=args.min_chars,
            decontaminator=decontaminator,
            seen_ids=seen_ids,
            removed=removed,
        )
        all_rows.extend(rows)
        print({"event": "source_done", "source": source, "rows": len(rows)}, flush=True)

    needle_rows = collect_multi_needle(
        target_count=args.multi_needle_count,
        rng=rng,
        min_chars=args.min_chars,
        decontaminator=decontaminator,
        seen_ids=seen_ids,
        removed=removed,
    )
    all_rows.extend(needle_rows)
    print({"event": "source_done", "source": "multi_needle", "rows": len(needle_rows)}, flush=True)

    train_rows, validation_rows = stratified_split(
        all_rows, args.validation_ratio, rng
    )
    train_ids = {row["doc_id"] for row in train_rows}
    validation_ids = {row["doc_id"] for row in validation_rows}
    overlap = train_ids & validation_ids
    if overlap:
        raise RuntimeError(f"Train/validation document overlap: {len(overlap)}")

    train_path = Path(args.train_output)
    validation_path = Path(args.validation_output)
    write_jsonl(train_path, train_rows, "train")
    write_jsonl(validation_path, validation_rows, "validation")

    metadata = {
        "seed": args.seed,
        "sources": SOURCE_FILES,
        "targets": {**targets, "multi_needle": args.multi_needle_count},
        "downloaded_files": downloaded,
        "longbench_files": decontaminator.files,
        "longbench_ngram_fingerprints": len(decontaminator.hashes),
        "decontamination": {
            "ngram_words": args.decontamination_ngram_words,
            "stride": args.decontamination_stride,
            "min_matches": args.decontamination_min_matches,
            "removed": dict(sorted(removed.items())),
        },
        "train_output": str(train_path),
        "validation_output": str(validation_path),
        "total_documents": len(all_rows),
        "train_documents": len(train_rows),
        "validation_documents": len(validation_rows),
        "train_source_counts": count_sources(train_rows),
        "validation_source_counts": count_sources(validation_rows),
        "train_length_buckets_chars": length_bucket_counts(train_rows),
        "validation_length_buckets_chars": length_bucket_counts(validation_rows),
        "document_overlap": 0,
    }
    metadata_path = Path(args.metadata_output)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(metadata, flush=True)


if __name__ == "__main__":
    main()
