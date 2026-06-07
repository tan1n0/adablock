from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_NEEDLES = (
    "The hidden access code is ADBLOCK-314.",
    "The secret project codename is BLUE LANTERN.",
    "The correct calibration number is 72941.",
    "The archived meeting location is Room K-17.",
    "The target checksum is 8F3C-11B9.",
)


FILLER_SENTENCES = (
    "This paragraph discusses routine background information and does not contain the requested fact.",
    "The document continues with operational details, examples, and contextual notes.",
    "Several unrelated entities are mentioned to make the retrieval task less trivial.",
    "The next section provides additional narrative material without changing the answer.",
    "Readers should distinguish local wording from the distant evidence inserted elsewhere.",
    "This part exists to increase context length and create distractor blocks.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare non-LongBench JSONL training prompts for AdaBlock oracle generation."
    )
    parser.add_argument("--output-jsonl", default="data/train_non_longbench.jsonl")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["qasper", "govreport", "needle"],
        choices=["qasper", "govreport", "narrativeqa", "needle"],
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples-per-source", type=int, default=200)
    parser.add_argument("--min-chars", type=int, default=1500)
    parser.add_argument("--max-chars", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--needle-samples", type=int, default=200)
    parser.add_argument("--needle-min-paragraphs", type=int, default=40)
    parser.add_argument("--needle-max-paragraphs", type=int, default=180)
    parser.add_argument(
        "--local-jsonl",
        default=None,
        help="Optional existing JSONL with a text field; appended as another source.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(clean_text(item) for item in value if item is not None)
    if isinstance(value, dict):
        return "\n".join(clean_text(item) for item in value.values() if item is not None)
    return " ".join(str(value).split())


def limit_chars(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()


def qasper_document_text(row: dict[str, Any]) -> str:
    full_text = row.get("full_text", {})
    sections = full_text.get("section_name", [])
    paragraphs = full_text.get("paragraphs", [])
    chunks: list[str] = []
    for section, section_paragraphs in zip(sections, paragraphs):
        section = clean_text(section)
        para = clean_text(section_paragraphs)
        if section:
            chunks.append(section)
        if para:
            chunks.append(para)
    return "\n\n".join(chunks) or clean_text(row.get("abstract", ""))


def qasper_answer_text(row: dict[str, Any]) -> str:
    answers = row.get("answers", [])
    extracted: list[str] = []
    for answer_group in answers:
        for answer in answer_group.get("answer", []) if isinstance(answer_group, dict) else []:
            if answer.get("unanswerable"):
                extracted.append("Unanswerable from the provided paper.")
            free_form = clean_text(answer.get("free_form_answer"))
            extractive = clean_text(answer.get("extractive_spans"))
            yes_no = answer.get("yes_no")
            if free_form:
                extracted.append(free_form)
            elif extractive:
                extracted.append(extractive)
            elif yes_no is not None:
                extracted.append("yes" if yes_no else "no")
    return extracted[0] if extracted else ""


def format_qasper(row: dict[str, Any], max_chars: int) -> str | None:
    question = clean_text(row.get("question"))
    document = qasper_document_text(row)
    answer = qasper_answer_text(row)
    if not question or not document:
        return None
    text = f"Question:\n{question}\n\nContext:\n{document}\n\nAnswer:\n{answer}"
    return limit_chars(text, max_chars)


def format_govreport(row: dict[str, Any], max_chars: int) -> str | None:
    document = clean_text(row.get("report") or row.get("document") or row.get("article"))
    summary = clean_text(row.get("summary") or row.get("highlights"))
    if not document:
        return None
    text = f"Document:\n{document}\n\nSummary:\n{summary}"
    return limit_chars(text, max_chars)


def format_narrativeqa(row: dict[str, Any], max_chars: int) -> str | None:
    document = clean_text(
        row.get("document", {}).get("text") if isinstance(row.get("document"), dict) else row.get("document")
    )
    question = clean_text(
        row.get("question", {}).get("text") if isinstance(row.get("question"), dict) else row.get("question")
    )
    answers = row.get("answers", [])
    if isinstance(answers, list) and answers:
        answer = clean_text(answers[0].get("text") if isinstance(answers[0], dict) else answers[0])
    else:
        answer = clean_text(answers)
    if not document or not question:
        return None
    text = f"Question:\n{question}\n\nContext:\n{document}\n\nAnswer:\n{answer}"
    return limit_chars(text, max_chars)


def import_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with `pip install -r requirement.txt`."
        ) from exc
    return load_dataset


def load_hf_source(source: str, split: str, max_samples: int, max_chars: int) -> Iterable[dict[str, str]]:
    load_dataset = import_datasets()
    if source == "qasper":
        dataset = load_dataset("allenai/qasper", split=split, trust_remote_code=True)
        formatter = format_qasper
    elif source == "govreport":
        dataset = load_dataset("ccdv/govreport-summarization", split=split)
        formatter = format_govreport
    elif source == "narrativeqa":
        dataset = load_dataset("deepmind/narrativeqa", split=split, trust_remote_code=True)
        formatter = format_narrativeqa
    else:
        raise ValueError(f"Unsupported Hugging Face source: {source}")

    emitted = 0
    for row in dataset:
        text = formatter(row, max_chars)
        if text:
            yield {"source": source, "text": text}
            emitted += 1
            if emitted >= max_samples:
                break


def make_needle_sample(rng: random.Random, index: int, min_paragraphs: int, max_paragraphs: int) -> dict[str, str]:
    paragraphs = rng.randint(min_paragraphs, max_paragraphs)
    needle = DEFAULT_NEEDLES[index % len(DEFAULT_NEEDLES)]
    insert_at = rng.randint(2, max(2, paragraphs - 3))

    body: list[str] = []
    for idx in range(paragraphs):
        if idx == insert_at:
            body.append(needle)
        else:
            body.append(rng.choice(FILLER_SENTENCES))

    question = "What is the hidden fact in the context?"
    text = f"Context:\n{chr(10).join(body)}\n\nQuestion:\n{question}\n\nAnswer:\n{needle}"
    return {"source": "needle", "text": text}


def load_needle_source(args: argparse.Namespace) -> Iterable[dict[str, str]]:
    rng = random.Random(args.seed)
    for idx in range(args.needle_samples):
        yield make_needle_sample(
            rng,
            idx,
            args.needle_min_paragraphs,
            args.needle_max_paragraphs,
        )


def load_local_jsonl(path: str | Path, max_chars: int) -> Iterable[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = clean_text(row.get("text") if isinstance(row, dict) else row)
            if text:
                yield {"source": "local", "text": limit_chars(text, max_chars)}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    total = 0
    with output_path.open("w", encoding="utf-8") as out:
        for source in args.sources:
            if source == "needle":
                rows = load_needle_source(args)
            else:
                rows = load_hf_source(
                    source,
                    split=args.split,
                    max_samples=args.max_samples_per_source,
                    max_chars=args.max_chars,
                )
            for row in rows:
                text = row["text"].strip()
                if len(text) < args.min_chars:
                    continue
                out.write(json.dumps({"text": text, "source": row["source"]}, ensure_ascii=False) + "\n")
                counts[row["source"]] = counts.get(row["source"], 0) + 1
                total += 1

        if args.local_jsonl:
            for row in load_local_jsonl(args.local_jsonl, args.max_chars):
                text = row["text"].strip()
                if len(text) < args.min_chars:
                    continue
                out.write(json.dumps({"text": text, "source": row["source"]}, ensure_ascii=False) + "\n")
                counts[row["source"]] = counts.get(row["source"], 0) + 1
                total += 1

    print({"output": str(output_path), "total": total, "counts": counts})


if __name__ == "__main__":
    main()
