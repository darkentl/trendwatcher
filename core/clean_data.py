"""Очистка новостных текстов и приведение источников к единому виду."""
import re
import html
import unicodedata
from urllib.parse import urlparse
import emoji
import pandas as pd


TG_GARBAGE_PATTERNS = [
    r"t\.me/\S+",
    r"Подписывайтесь.*",
    r"Наш канал.*",
]


def _safe_text(value):
    if value is None or pd.isna(value):
        return ""
    return str(value)


def normalize_telegram_formatting(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"~~(.*?)~~", r"\1", text)
    # Убираем только маркеры списков/цитат, не съедая кавычки в заголовках.
    text = re.sub(r"^\s*[>•●▪▫◦\-*–—]+\s*", "", text, flags=re.MULTILINE)
    return text


def collapse_lines(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = " ".join(lines)
    text = re.sub(r"\.\s+", ".\n", text)
    return text


def normalize_for_dedup(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def deduplicate_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    seen = set()
    unique = []

    for sent in sentences:
        key = normalize_for_dedup(sent)
        if key and key not in seen:
            seen.add(key)
            unique.append(sent.strip())

    return " ".join(unique)


def deduplicate_paragraphs(text):
    paragraphs = text.split("\n")
    seen = set()
    unique = []

    for para in paragraphs:
        key = normalize_for_dedup(para)
        if key and key not in seen:
            seen.add(key)
            unique.append(para.strip())

    return "\n".join(unique)


def clean_text(text):
    text = _safe_text(text)
    if not text:
        return ""

    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    if emoji is not None:
        text = emoji.replace_emoji(text, replace="")
    text = normalize_telegram_formatting(text)

    for pattern in TG_GARBAGE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = collapse_lines(text)

    text = deduplicate_sentences(text)
    text = deduplicate_paragraphs(text)

    return text.strip()


def normalize_source_display(source_type, source_value):
    source_type = _safe_text(source_type).lower()
    source_value = _safe_text(source_value).strip()

    if not source_value:
        return ""

    if source_type == "telegram":
        return source_value.replace("@", "")

    if "://" not in source_value:
        source_value = "https://" + source_value

    parsed = urlparse(source_value)
    return (parsed.netloc or parsed.path).lower().replace("www.", "")


def run_clean(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    required = {"text", "url", "source_type", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В {input_csv} нет обязательных колонок: {sorted(missing)}")

    if df.empty:
        df.assign(source=[]).to_csv(output_csv, index=False)
        print(f"[CLEAN] Очищено строк: 0. Сохранено: {output_csv}")
        return

    df["text"] = df["text"].apply(clean_text)
    df["source"] = df.apply(
        lambda row: normalize_source_display(row["source_type"], row["source"]),
        axis=1,
    )

    df = df[df["text"].str.len() > 0].copy()
    df.to_csv(output_csv, index=False)

    print(f"[CLEAN] Очищено строк: {len(df)}. Сохранено: {output_csv}")
