"""Отсекает инфоповоды, которые уже были обработаны ранее."""

import hashlib
import json
import os

import pandas as pd

from config import SEEN_INFOPVODS_JSON


def text_hash(text):
    """Создаёт стабильный хэш текста для истории обработанных новостей."""
    return hashlib.md5(str(text or "").encode("utf-8")).hexdigest()


def load_seen():
    """Загружает хэши уже обработанных инфоповодов."""
    if not os.path.exists(SEEN_INFOPVODS_JSON):
        return set()
    with open(SEEN_INFOPVODS_JSON, "r", encoding="utf-8") as f:
        return set(json.load(f))


def save_seen(seen):
    """Сохраняет обновлённую историю обработанных инфоповодов."""
    os.makedirs(os.path.dirname(SEEN_INFOPVODS_JSON) or ".", exist_ok=True)
    with open(SEEN_INFOPVODS_JSON, "w", encoding="utf-8") as f:
        json.dump(sorted(seen), f, ensure_ascii=False, indent=2)


def filter_new_infopovods(csv_path):
    """Возвращает только новые строки и сразу обновляет историю."""
    df = pd.read_csv(csv_path)
    if df.empty:
        return df

    seen = load_seen()
    df["__hash"] = df["text"].apply(text_hash)
    new_df = df[~df["__hash"].isin(seen)].copy()

    seen.update(new_df["__hash"].tolist())
    save_seen(seen)

    return new_df.drop(columns="__hash")
