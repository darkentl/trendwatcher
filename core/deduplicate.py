"""Дедубликация новостей по текстовой близости и embedding-сходству."""
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
import hashlib
import json
import os
import re
from threading import Lock

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .llm import get_embedding
from config import (
    DEDUP_WORKERS,
    DEDUP_USE_EMBEDDINGS,
    DEDUP_LEVEL_1_THRESHOLD,
    DEDUP_LEVEL_2_THRESHOLD,
    DEDUP_TEXT_JACCARD_THRESHOLD,
    DEDUP_TEXT_SEQUENCE_THRESHOLD,
    DEDUP_TITLE_SEQUENCE_THRESHOLD,
    DATA_DIR,
)

# Дедубликация: сначала быстрые текстовые правила, embeddings только при необходимости.

DEDUP_MODE = os.getenv(
    "DEDUP_MODE",
    "adaptive" if DEDUP_USE_EMBEDDINGS else "text",
).strip().lower()

# Экономный adaptive-режим: сужаем серую зону, чтобы реже считать embeddings.
# >= dup_threshold — считаем дублем без embedding, <= new_threshold — считаем новой новостью без embedding.
DEDUP_CHEAP_DUP_THRESHOLD = float(os.getenv("DEDUP_CHEAP_DUP_THRESHOLD", "0.78"))
DEDUP_CHEAP_NEW_THRESHOLD = float(os.getenv("DEDUP_CHEAP_NEW_THRESHOLD", "0.50"))
DEDUP_ADAPTIVE_EMBEDDING_THRESHOLD = float(
    os.getenv("DEDUP_ADAPTIVE_EMBEDDING_THRESHOLD", str(DEDUP_LEVEL_2_THRESHOLD))
)
DEDUP_SAVE_SOURCES_COUNT = os.getenv("DEDUP_SAVE_SOURCES_COUNT", "1") == "1"

_DEFAULT_CACHE_PATH = Path(DATA_DIR) / "embedding_cache.json"
DEDUP_EMBEDDING_CACHE_PATH = Path(
    os.getenv("DEDUP_EMBEDDING_CACHE_PATH", str(_DEFAULT_CACHE_PATH))
)


RUSSIAN_STOPWORDS = set("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы
по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг
ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом
себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам
чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого
какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда
зачем всех никогда можно при наконец два об другой хоть после над больше тот
через эти нас про всего них какая много разве три эту моя впрочем хорошо свою
этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между
""".split())

WORD_ENDINGS = [
    "иями", "ями", "ами", "ого", "его", "ому", "ему", "ыми", "ими",
    "ость", "ости", "иться", "ться", "ать", "ять", "лась", "лись",
    "лся", "ой", "ей", "ый", "ий", "ая", "яя", "ое", "ее", "ые",
    "ие", "ам", "ям", "ах", "ях", "ов", "ев", "ом", "ем", "ою", "ею",
]

IMPORTANT_ENTITY_PATTERNS = [
    "цб", "банк россии", "центробанк", "автоваз", "лада", "lada",
    "утильсбор", "осаго", "каско", "автокредит", "автокредиты",
    "господдержка", "субсид", "дилер", "дилеры", "китай", "электромоб",
    "маркетплейс", "авито", "авто.ру", "autoru", "яндекс", "сбер",
    "втб", "тинькофф", "альфа", "газпромбанк", "росбанк",
]

_cache_lock = Lock()
_embedding_cache = None


def normalize_word(word):
    word = str(word or "").lower().replace("ё", "е")
    for ending in WORD_ENDINGS:
        if len(word) > 5 and word.endswith(ending):
            return word[:-len(ending)]
    return word


def normalized_tokens(text):
    words = re.findall(r"[а-яa-z0-9]+", str(text or "").lower().replace("ё", "е"))
    tokens = []
    for word in words:
        if len(word) < 3 or word in RUSSIAN_STOPWORDS:
            continue
        tokens.append(normalize_word(word))
    return tokens


def text_similarity(a, b):
    tokens_a = set(normalized_tokens(a))
    tokens_b = set(normalized_tokens(b))
    if not tokens_a or not tokens_b:
        return 0, 0

    jaccard = len(tokens_a & tokens_b) / max(1, len(tokens_a | tokens_b))
    sequence = SequenceMatcher(None, " ".join(sorted(tokens_a)), " ".join(sorted(tokens_b))).ratio()
    return jaccard, sequence


def title_similarity(a, b):
    tokens_a = normalized_tokens(a)
    tokens_b = normalized_tokens(b)
    if not tokens_a or not tokens_b:
        return 0
    return SequenceMatcher(None, " ".join(tokens_a), " ".join(tokens_b)).ratio()


def extract_entities(text):
    """Дешевое извлечение сущностей без LLM: бренды, регуляторы, банки, авто-термины."""
    low = str(text or "").lower().replace("ё", "е")
    found = set()

    for pattern in IMPORTANT_ENTITY_PATTERNS:
        if pattern in low:
            found.add(pattern)

    # Латиница/цифры вроде Lada, Haval, Chery, Omoda, XC90, H6.
    for item in re.findall(r"\b[a-zA-Z][a-zA-Z0-9.-]{2,}\b", str(text or "")):
        found.add(item.lower())

    # Частые русские имена собственные/бренды. Не идеально, зато дешево.
    for item in re.findall(r"\b[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,})?\b", str(text or "")):
        normalized = item.lower().replace("ё", "е")
        if normalized not in RUSSIAN_STOPWORDS:
            found.add(normalized)

    return found


def entity_overlap(a, b):
    entities_a = extract_entities(a)
    entities_b = extract_entities(b)
    if not entities_a or not entities_b:
        return 0
    return len(entities_a & entities_b) / max(1, min(len(entities_a), len(entities_b)))


def date_proximity(date_a, date_b):
    """1.0 — в тот же час/день, 0.0 — даты нет или слишком далеко."""
    try:
        a = pd.to_datetime(date_a, errors="coerce", utc=True)
        b = pd.to_datetime(date_b, errors="coerce", utc=True)
        if pd.isna(a) or pd.isna(b):
            return 0
        hours = abs((a - b).total_seconds()) / 3600
        if hours <= 6:
            return 1.0
        if hours <= 24:
            return 0.8
        if hours <= 72:
            return 0.45
        if hours <= 168:
            return 0.2
        return 0
    except Exception:
        return 0


def calc_cheap_similarity(row_a, row_b):
    title_a = row_a.get("title", "")
    title_b = row_b.get("title", "")
    text_a = f"{title_a}. {row_a.get('text', '')}"
    text_b = f"{title_b}. {row_b.get('text', '')}"

    jaccard, sequence = text_similarity(text_a, text_b)
    title_seq = title_similarity(title_a, title_b)
    entities = entity_overlap(text_a, text_b)
    dates = date_proximity(row_a.get("date", ""), row_b.get("date", ""))

    # Веса помогают принять больше решений без платных embedding-вызовов.
    score = (
        title_seq * 0.35 +
        jaccard * 0.25 +
        sequence * 0.20 +
        entities * 0.15 +
        dates * 0.05
    )

    return min(1.0, max(0.0, score))


def looks_same_by_text(row_a, row_b):
    title_a = row_a.get("title", "")
    title_b = row_b.get("title", "")
    text_a = f"{title_a}. {row_a.get('text', '')}"
    text_b = f"{title_b}. {row_b.get('text', '')}"

    jaccard, sequence = text_similarity(text_a, text_b)
    title_seq = title_similarity(title_a, title_b)

    if jaccard >= DEDUP_TEXT_JACCARD_THRESHOLD:
        return True
    if sequence >= DEDUP_TEXT_SEQUENCE_THRESHOLD:
        return True
    if title_seq >= DEDUP_TITLE_SEQUENCE_THRESHOLD and jaccard >= DEDUP_TEXT_JACCARD_THRESHOLD * 0.65:
        return True

    return False


def merge_clusters_by_text(df, clusters):
    if not clusters:
        return []

    parent = list(range(len(clusters)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    representatives = []
    for cluster in clusters:
        representatives.append(df.iloc[cluster[0]])

    for i in range(len(representatives)):
        for j in range(i + 1, len(representatives)):
            if looks_same_by_text(representatives[i], representatives[j]):
                union(i, j)

    merged_by_root = {}
    for cluster_index, cluster in enumerate(clusters):
        root = find(cluster_index)
        merged_by_root.setdefault(root, [])
        merged_by_root[root].extend(cluster)

    return list(merged_by_root.values())


def extract_core(text):
    text = str(text or "")
    parts = [p.strip() for p in text.split("\n") if len(p.strip()) > 40]
    return "\n".join(parts[:5]) or text[:1000]


def build_clusters(embeddings, threshold):
    if len(embeddings) == 0:
        return []

    sim_matrix = cosine_similarity(embeddings)
    visited = set()
    clusters = []

    for i in range(len(embeddings)):
        if i in visited:
            continue

        stack = [i]
        cluster = []

        while stack:
            idx = stack.pop()
            if idx in visited:
                continue

            visited.add(idx)
            cluster.append(idx)

            similar_idxs = np.where(sim_matrix[idx] > threshold)[0]
            for s in similar_idxs:
                if s not in visited:
                    stack.append(int(s))

        clusters.append(cluster)

    return clusters


def source_priority(row):
    url = str(row.get("url", "")).lower()

    if "cbr.ru" in url:
        return 1

    if any(x in url for x in [
        "motor.ru", "gazeta.ru", "autostat.ru", "kolesa.ru",
        "110km.ru", "carexpo.ru", "mail.ru", "drom.ru",
        "5koleso.ru", "ixbt.com", "3dnews.ru"
    ]):
        return 2

    if "t.me" in url:
        return 3

    return 4


def _normalized_for_hash(text):
    tokens = normalized_tokens(extract_core(text))
    return " ".join(tokens[:400])


def _embedding_cache_key(text):
    normalized = _normalized_for_hash(text)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def _load_embedding_cache():
    global _embedding_cache
    if _embedding_cache is not None:
        return _embedding_cache

    with _cache_lock:
        if _embedding_cache is not None:
            return _embedding_cache
        try:
            if DEDUP_EMBEDDING_CACHE_PATH.exists():
                with open(DEDUP_EMBEDDING_CACHE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                _embedding_cache = data if isinstance(data, dict) else {}
            else:
                _embedding_cache = {}
        except Exception:
            _embedding_cache = {}
        return _embedding_cache


def _save_embedding_cache():
    with _cache_lock:
        try:
            DEDUP_EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = DEDUP_EMBEDDING_CACHE_PATH.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(_embedding_cache or {}, f, ensure_ascii=False)
            tmp_path.replace(DEDUP_EMBEDDING_CACHE_PATH)
        except Exception as exc:
            print(f"[DEDUP] Не удалось сохранить embedding-cache: {exc}")


def get_or_create_embedding(text):
    cache = _load_embedding_cache()
    key = _embedding_cache_key(text)

    cached = cache.get(key)
    if cached:
        return cached

    embedding = get_embedding(text)
    if embedding is None:
        raise RuntimeError("get_embedding вернул None")

    with _cache_lock:
        cache[key] = embedding
    return embedding


def _cosine_one(a, b):
    arr = np.vstack([a, b])
    return float(cosine_similarity(arr)[0, 1])


def _full_embedding_clusters(df):
    """Старый режим: считаем embeddings для всех строк."""
    with ThreadPoolExecutor(max_workers=DEDUP_WORKERS) as ex:
        df["embedding"] = list(ex.map(get_or_create_embedding, df["core_text"].tolist()))

    _save_embedding_cache()

    embeddings = np.vstack(df["embedding"].values)

    clusters_lvl1 = build_clusters(embeddings, threshold=DEDUP_LEVEL_1_THRESHOLD)

    rep_indices = [c[0] for c in clusters_lvl1]
    rep_embeddings = embeddings[rep_indices]

    clusters_lvl2 = build_clusters(rep_embeddings, threshold=DEDUP_LEVEL_2_THRESHOLD)

    final_clusters = []
    for cluster in clusters_lvl2:
        merged = []
        for rep_idx in cluster:
            merged.extend(clusters_lvl1[rep_idx])
        final_clusters.append(merged)

    return final_clusters


def _adaptive_clusters(df):
    """Склеивает новости: правилами для очевидных пар, embeddings для спорных."""
    n = len(df)
    parent = list(range(n))
    embedding_calls = 0
    cache_hits_or_reuses = 0
    cheap_duplicates = 0
    cheap_skips = 0
    uncertain_pairs = 0

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    local_embeddings = {}

    def emb_for_idx(idx):
        nonlocal embedding_calls, cache_hits_or_reuses
        if idx in local_embeddings:
            cache_hits_or_reuses += 1
            return local_embeddings[idx]

        text = df.iloc[idx].get("core_text", "")
        key = _embedding_cache_key(text)
        cache = _load_embedding_cache()
        already_cached = key in cache
        emb = get_or_create_embedding(text)
        local_embeddings[idx] = emb
        if already_cached:
            cache_hits_or_reuses += 1
        else:
            embedding_calls += 1
        return emb

    for i in range(n):
        row_i = df.iloc[i]
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue

            row_j = df.iloc[j]
            cheap_score = calc_cheap_similarity(row_i, row_j)

            if cheap_score >= DEDUP_CHEAP_DUP_THRESHOLD:
                union(i, j)
                cheap_duplicates += 1
                continue

            if cheap_score <= DEDUP_CHEAP_NEW_THRESHOLD:
                cheap_skips += 1
                continue

            uncertain_pairs += 1
            emb_i = emb_for_idx(i)
            emb_j = emb_for_idx(j)
            sim = _cosine_one(emb_i, emb_j)

            if sim >= DEDUP_ADAPTIVE_EMBEDDING_THRESHOLD:
                union(i, j)

    _save_embedding_cache()

    clusters_by_root = {}
    for idx in range(n):
        root = find(idx)
        clusters_by_root.setdefault(root, []).append(idx)

    print(
        "[DEDUP] Adaptive: "
        f"cheap-дубли={cheap_duplicates}, "
        f"пропуски={cheap_skips}, "
        f"спорные пары={uncertain_pairs}, "
        f"embedding-вызовы={embedding_calls}, "
        f"cache/reuse={cache_hits_or_reuses}"
    )

    return list(clusters_by_root.values())


def _add_cluster_metadata(cluster_df):
    best_row = cluster_df.sort_values("priority").iloc[0].copy()

    if DEDUP_SAVE_SOURCES_COUNT:
        sources = []
        urls = []
        for _, row in cluster_df.iterrows():
            source = str(row.get("source", "") or "").strip()
            url = str(row.get("url", "") or "").strip()
            if source and source not in sources:
                sources.append(source)
            if url and url not in urls:
                urls.append(url)

        best_row["sources_count"] = len(urls) if urls else len(cluster_df)
        best_row["duplicate_count"] = len(cluster_df)
        best_row["all_sources"] = "; ".join(sources)
        best_row["all_urls"] = "; ".join(urls)

    return best_row


def run_deduplicate(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    if df.empty:
        df.to_csv(output_csv, index=False)
        print(f"[DEDUP] Входной файл пуст. Сохранено: {output_csv}")
        return

    if "text" not in df.columns:
        raise ValueError(f"В {input_csv} нет обязательной колонки text")

    df = df[df["text"].fillna("").astype(str).str.strip().ne("")].copy()
    if df.empty:
        df.to_csv(output_csv, index=False)
        print(f"[DEDUP] После фильтрации пустых текстов строк нет. Сохранено: {output_csv}")
        return

    df = df.reset_index(drop=True)
    df["core_text"] = df["text"].apply(extract_core)

    mode = DEDUP_MODE
    if not DEDUP_USE_EMBEDDINGS and mode in {"adaptive", "full", "embeddings", "embedding"}:
        mode = "text"

    if mode in {"full", "embeddings", "embedding"}:
        print("[DEDUP] Режим: full embeddings")
        final_clusters = _full_embedding_clusters(df)
    elif mode == "adaptive":
        print(f"[DEDUP] Режим: adaptive; embeddings только для спорных пар "
              f"({DEDUP_CHEAP_NEW_THRESHOLD}<score<{DEDUP_CHEAP_DUP_THRESHOLD})")
        final_clusters = _adaptive_clusters(df)
    else:
        print("[DEDUP] Режим: text; embeddings отключены")
        final_clusters = [[i] for i in range(len(df))]

    # Финальный текстовый проход склеивает очевидные дубли из разных кластеров.
    final_clusters = merge_clusters_by_text(df, final_clusters)

    primary_rows = []

    for cluster in final_clusters:
        cluster_df = df.iloc[cluster].copy()
        cluster_df["priority"] = cluster_df.apply(source_priority, axis=1)
        primary_rows.append(_add_cluster_metadata(cluster_df))

    dedup_df = pd.DataFrame(primary_rows).drop(
        columns=["priority", "embedding", "core_text"], errors="ignore"
    )

    dedup_df.to_csv(output_csv, index=False)
    print(f"[DEDUP] Было {len(df)}, осталось {len(dedup_df)}. Сохранено: {output_csv}")
