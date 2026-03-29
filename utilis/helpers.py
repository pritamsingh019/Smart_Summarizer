import re
from pathlib import Path
from typing import Iterable

import pandas as pd

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
MAX_DATAFRAME_ROWS = 50_000
TEXT_CHUNK_SIZE = 4_000

FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def infer_file_type(filename: str) -> str:
    extension = Path(filename or "").suffix.lower()
    mapping = {
        ".csv": "csv",
        ".xlsx": "excel",
        ".xls": "excel",
        ".pdf": "pdf",
        ".txt": "txt",
    }
    return mapping.get(extension, "unknown")


def validate_file_bytes(raw_bytes: bytes) -> str | None:
    if raw_bytes is None or len(raw_bytes) == 0:
        return "The uploaded file is empty."
    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        return "The uploaded file exceeds the 10MB size limit."
    return None


def normalise_whitespace(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text or "").strip()


def compact_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sentence_split(text: str) -> list[str]:
    clean_text = compact_text(text)
    if not clean_text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", clean_text)
    return [sentence.strip() for sentence in sentences if sentence and re.search(r"\w", sentence)]


def chunk_text(text: str, chunk_size: int = TEXT_CHUNK_SIZE) -> list[str]:
    sentences = sentence_split(text)
    if not sentences:
        clean_text = compact_text(text)
        return [clean_text] if clean_text else []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_size = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_sentences and current_size + sentence_length > chunk_size:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_size = sentence_length
        else:
            current_sentences.append(sentence)
            current_size += sentence_length

    if current_sentences:
        chunks.append(" ".join(current_sentences))
    return chunks


def safe_percentage_change(start_value: float | int | None, end_value: float | int | None) -> float | None:
    if start_value is None or end_value is None:
        return None
    if pd.isna(start_value) or pd.isna(end_value):
        return None
    if float(start_value) == 0.0:
        return None
    return ((float(end_value) - float(start_value)) / abs(float(start_value))) * 100.0


def normalise_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def tokenise_label(value: object) -> list[str]:
    return [token for token in normalise_key(value).split() if token]


def humanize_label(value: object) -> str:
    tokens = tokenise_label(value)
    if not tokens:
        return str(value)
    return " ".join(tokens).title()


def deduplicate_columns(columns: Iterable[object]) -> list[str]:
    seen: dict[str, int] = {}
    deduplicated: list[str] = []

    for index, column in enumerate(columns, start=1):
        base_name = str(column).strip() or f"column_{index}"
        seen[base_name] = seen.get(base_name, 0) + 1
        if seen[base_name] == 1:
            deduplicated.append(base_name)
        else:
            deduplicated.append(f"{base_name}_{seen[base_name]}")
    return deduplicated


def coerce_possible_datetime(series: pd.Series, threshold: float = 0.7) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return series
    parsed = pd.to_datetime(series, errors="coerce")
    if len(series) == 0:
        return series
    if parsed.notna().mean() >= threshold:
        return parsed
    return series


def format_metric(value: object, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    numeric_value = float(value)
    abs_value = abs(numeric_value)
    if abs_value >= 1_000_000:
        return f"{numeric_value / 1_000_000:.{decimals}f}M"
    if abs_value >= 1_000:
        return f"{numeric_value / 1_000:.{decimals}f}K"
    if numeric_value.is_integer():
        return f"{int(numeric_value):,}"
    return f"{numeric_value:,.{decimals}f}"
