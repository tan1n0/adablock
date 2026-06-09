from __future__ import annotations

import json
import re
import string
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LONG_INPUT_SHORT_OUTPUT_TASKS = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "musique",
    "gov_report",
)


TASK_MAX_NEW_TOKENS = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "musique": 32,
    "gov_report": 512,
}


TASK_PROMPTS = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie script, "
        "and a question. Answer the question as concisely as you can, using a "
        "single phrase if possible. Do not provide any explanation.\n\n"
        "Story: {context}\n\nQuestion: {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the question "
        "as concisely as you can, using a single phrase or sentence if possible. "
        "If the question cannot be answered based on the information in the "
        "article, write \"unanswerable\". If the question is a yes/no question, "
        "answer \"yes\", \"no\", or \"unanswerable\". Do not provide any "
        "explanation.\n\nArticle: {context}\n\nQuestion: {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\n"
        "Now, answer the following question based on the above text, only give "
        "me the answer and do not output any other words.\n\n"
        "Question: {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Output only the final "
        "short answer. Do not explain your reasoning. Do not say that the "
        "passages do not contain the answer. If multiple passages are needed, "
        "combine them silently and still output only the answer.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Question: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Output only the final "
        "short answer. Do not explain your reasoning. Do not say that the "
        "passages do not contain the answer. If multiple passages are needed, "
        "combine them silently and still output only the answer.\n\n"
        "The following are given passages.\n{context}\n\n"
        "Question: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page "
        "summary of the report.\n\nReport:\n{context}\n\nSummary:"
    ),
}


def load_dtype(name: str):
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def normalize_device_map(value: str | None) -> str | None:
    if value is None:
        return None
    if value.lower() in {"none", "null", "false"}:
        return None
    return value


def load_longbench_task(task: str, split: str = "test", cache_dir: str | Path | None = None):
    if split != "test":
        raise ValueError(
            "LongBench v1 public files are test JSONL files. Use --local-data-dir for custom splits."
        )
    try:
        from huggingface_hub import hf_hub_download
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub/datasets. Install with `pip install -r requirement.txt`."
        ) from exc
    cache_root = Path(cache_dir) if cache_dir else Path(".cache") / "longbench"
    extracted_root = cache_root / "extracted"
    local_candidates = [
        extracted_root / f"{task}.jsonl",
        extracted_root / "data" / f"{task}.jsonl",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return load_local_task(candidate)

    download_errors: list[str] = []
    try:
        path = hf_hub_download(
            repo_id="zai-org/LongBench",
            repo_type="dataset",
            filename=f"{task}/test-00000-of-00001.parquet",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        return list(load_dataset("parquet", data_files=path, split="train"))
    except Exception as exc:
        download_errors.append(f"parquet: {exc}")

    try:
        zip_path = hf_hub_download(
            repo_id="zai-org/LongBench",
            repo_type="dataset",
            filename="data.zip",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        extracted_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extracted_root)
        for candidate in local_candidates:
            if candidate.exists():
                return load_local_task(candidate)
        raise FileNotFoundError(f"Could not find {task}.jsonl after extracting {zip_path}")
    except Exception as exc:
        download_errors.append(f"data.zip: {exc}")

    try:
        path = hf_hub_download(
            repo_id="THUDM/LongBench",
            repo_type="dataset",
            filename=f"data/{task}.jsonl",
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        return load_local_task(path)
    except Exception as exc:
        download_errors.append(f"legacy jsonl: {exc}")
        raise RuntimeError(
            f"Failed to load LongBench task {task}. Tried parquet, data.zip, and legacy JSONL. "
            f"Errors: {' | '.join(download_errors)}"
        ) from exc


def load_local_task(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def format_longbench_prompt(task: str, row: dict[str, Any]) -> str:
    if task not in TASK_PROMPTS:
        raise ValueError(f"Unsupported LongBench task: {task}")
    return TASK_PROMPTS[task].format(
        context=row.get("context", ""),
        input=row.get("input", row.get("question", "")),
    )


def build_model_input_text(tokenizer, prompt: str, use_chat_template: bool = True) -> str:
    if not use_chat_template:
        return prompt
    if getattr(tokenizer, "chat_template", None) is None:
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def truncate_middle(input_ids, max_length: int):
    if input_ids.shape[-1] <= max_length:
        return input_ids
    half = max_length // 2
    return input_ids[..., :half].new_tensor(
        input_ids[..., :half].tolist()[0]
        + input_ids[..., -(max_length - half) :].tolist()[0]
    ).unsqueeze(0)


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return " ".join(remove_articles(remove_punc(text.lower())).split())


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_length(a: list[str], b: list[str]) -> int:
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for idx_b, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[idx_b - 1] + 1)
            else:
                current.append(max(previous[idx_b], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    lcs = lcs_length(pred_tokens, truth_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def score_prediction(task: str, prediction: str, answers: Iterable[str]) -> float:
    answer_list = [str(answer) for answer in answers]
    if not answer_list:
        return 0.0
    if task == "gov_report":
        return max(rouge_l_f1(prediction, answer) for answer in answer_list) * 100.0
    return max(token_f1(prediction, answer) for answer in answer_list) * 100.0


def extract_answers(row: dict[str, Any]) -> list[str]:
    answers = row.get("answers", row.get("answer", []))
    if isinstance(answers, str):
        return [answers]
    if isinstance(answers, list):
        return [str(answer) for answer in answers]
    return [str(answers)]


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)
