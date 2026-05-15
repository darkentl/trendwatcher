"""Фильтр релевантности: правила сначала, LLM только для спорных текстов."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List

import pandas as pd

from config import (
    FILTER_MODEL,
    LLM_FILTER_KEYWORD_MIN_SCORE,
    LLM_FILTER_WORKERS,
    USE_LLM_FILTER,
    FILTER_UNKNOWN_WITH_LLM,
)
from .llm import (
    FILTER_PROMPT,
    ask_llm,
    extract_json,
    is_relevant,
    keyword_relevance_score,
)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(str(value).strip())
    except ValueError:
        print(f"[FILTER] Некорректное значение {name}={value!r}. Использую {default}.")
        return default


def _filter_mode() -> str:
    """Выбирает режим фильтра с поддержкой старых настроек."""
    explicit = os.getenv("LLM_FILTER_MODE")
    if explicit:
        mode = explicit.strip().lower()
    elif USE_LLM_FILTER:
        mode = "llm"
    elif FILTER_UNKNOWN_WITH_LLM:
        mode = "adaptive"
    else:
        mode = "adaptive"

    aliases = {
        "adaptive": "adaptive",
        "hybrid": "adaptive",
        "combined": "adaptive",
        "комбинированный": "adaptive",
        "keyword": "keyword",
        "keywords": "keyword",
        "cheap": "keyword",
        "rules": "keyword",
        "llm": "llm",
        "full_llm": "llm",
        "all_llm": "llm",
    }
    return aliases.get(mode, "adaptive")


FILTER_MODE = _filter_mode()

# Экономные пороги: больше решений принимается keyword-фильтром без LLM.
AUTO_ACCEPT_SCORE = _env_int(
    "LLM_FILTER_AUTO_ACCEPT_SCORE",
    max(int(LLM_FILTER_KEYWORD_MIN_SCORE) + 1, 3),
)
AUTO_REJECT_SCORE = _env_int("LLM_FILTER_AUTO_REJECT_SCORE", 1)
SAVE_DEBUG = _env_bool("LLM_FILTER_SAVE_DEBUG", False)


@dataclass
class RelevanceResult:
    relevant: bool
    method: str
    score: int
    reasons: List[str]
    llm_used: bool = False
    error: str = ""


def _make_text(row: pd.Series) -> str:
    title = str(row.get("title", "") or "").strip()
    text = str(row.get("text", "") or "").strip()
    if title:
        return f"{title}. {text}".strip()
    return text


def _ask_llm_relevance(text: str) -> bool:
    content = ask_llm(
        model=FILTER_MODEL,
        system=FILTER_PROMPT,
        user=text,
        temp=0,
    )
    data = extract_json(content)
    if not isinstance(data, dict):
        raise ValueError("LLM returned JSON, but it is not an object")
    return bool(data.get("relevant", False))


def _keyword_only(text: str) -> RelevanceResult:
    score, reasons = keyword_relevance_score(text)
    relevant = bool(score >= LLM_FILTER_KEYWORD_MIN_SCORE)
    return RelevanceResult(
        relevant=relevant,
        method="keyword_accept" if relevant else "keyword_reject",
        score=int(score),
        reasons=list(reasons),
        llm_used=False,
    )


def _llm_mode(text: str) -> RelevanceResult:
    """Полный LLM-режим без переключения на запасные модели."""
    score, reasons = keyword_relevance_score(text)
    relevant = bool(is_relevant(text))
    return RelevanceResult(
        relevant=relevant,
        method="llm",
        score=int(score),
        reasons=list(reasons),
        llm_used=True,
    )


def _adaptive(text: str) -> RelevanceResult:
    score, reasons = keyword_relevance_score(text)
    score = int(score)
    reasons = list(reasons)

    # Явно релевантное пропускаем без LLM.
    if score >= AUTO_ACCEPT_SCORE:
        return RelevanceResult(
            relevant=True,
            method="adaptive_keyword_accept",
            score=score,
            reasons=reasons,
            llm_used=False,
        )

    # Явно нерелевантное отклоняем без LLM.
    if score <= AUTO_REJECT_SCORE:
        return RelevanceResult(
            relevant=False,
            method="adaptive_keyword_reject",
            score=score,
            reasons=reasons,
            llm_used=False,
        )

    # Только оставшуюся узкую серую зону проверяет LLM.
    try:
        relevant = _ask_llm_relevance(text)
        return RelevanceResult(
            relevant=bool(relevant),
            method="adaptive_llm_gray_zone",
            score=score,
            reasons=reasons,
            llm_used=True,
        )
    except Exception as e:
        # При ошибке LLM возвращаемся к keyword-порогу.
        fallback = bool(score >= LLM_FILTER_KEYWORD_MIN_SCORE)
        return RelevanceResult(
            relevant=fallback,
            method="adaptive_llm_error_keyword_decision",
            score=score,
            reasons=reasons,
            llm_used=True,
            error=f"{type(e).__name__}: {e}",
        )


def _safe_filter(text: str) -> RelevanceResult:
    try:
        if FILTER_MODE == "keyword":
            return _keyword_only(text)
        if FILTER_MODE == "llm":
            return _llm_mode(text)
        return _adaptive(text)
    except Exception as e:
        # Один проблемный текст не должен останавливать весь пайплайн.
        try:
            score, reasons = keyword_relevance_score(text)
            fallback = bool(score >= LLM_FILTER_KEYWORD_MIN_SCORE)
            return RelevanceResult(
                relevant=fallback,
                method="fatal_error_keyword_decision",
                score=int(score),
                reasons=list(reasons),
                llm_used=False,
                error=f"{type(e).__name__}: {e}",
            )
        except Exception as inner:
            return RelevanceResult(
                relevant=False,
                method="fatal_error_reject",
                score=0,
                reasons=[],
                llm_used=False,
                error=f"{type(e).__name__}: {e}; ошибка резервного решения: {type(inner).__name__}: {inner}",
            )


def run_llm_filter(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    if df.empty:
        df.to_csv(output_csv, index=False)
        print(f"[FILTER] Готово: релевантных 0/0. Сохранено: {output_csv}")
        return

    if "text" not in df.columns:
        raise ValueError(f"В {input_csv} нет обязательной колонки text")

    texts = [_make_text(row) for _, row in df.iterrows()]
    results: List[RelevanceResult | None] = [None] * len(texts)

    workers = max(1, int(LLM_FILTER_WORKERS))
    llm_calls = 0

    print(
        f"[FILTER] Режим: {FILTER_MODE}; "
        f"автопринятие от {AUTO_ACCEPT_SCORE}, автоотклонение до {AUTO_REJECT_SCORE}"
    )
    print(f"[FILTER] К обработке: {len(texts)} строк, потоков: {workers}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_safe_filter, text): i for i, text in enumerate(texts)}
        for done, future in enumerate(as_completed(futures), 1):
            i = futures[future]
            result = future.result()
            results[i] = result
            if result.llm_used:
                llm_calls += 1
            print(
                f"[FILTER] Обработка: {done}/{len(texts)}, "
                f"метод: {result.method}, "
                f"LLM-вызовов: {llm_calls}",
                end="\r",
            )
    print()

    # На случай, если какой-то поток не вернул результат.
    final_results: List[RelevanceResult] = [
        r if r is not None else RelevanceResult(False, "missing_result_reject", 0, [])
        for r in results
    ]

    df["relevant"] = [r.relevant for r in final_results]

    if SAVE_DEBUG:
        df["relevance_score"] = [r.score for r in final_results]
        df["relevance_method"] = [r.method for r in final_results]
        df["relevance_reasons"] = [", ".join(r.reasons[:12]) for r in final_results]
        df["relevance_llm_used"] = [r.llm_used for r in final_results]
        df["relevance_error"] = [r.error for r in final_results]

    auto_df = df[df["relevant"]].copy()

    auto_df = auto_df.drop(columns=["relevant"])

    auto_df.to_csv(output_csv, index=False)

    accepted_by_keywords = sum(
        1 for r in final_results if r.method == "adaptive_keyword_accept"
    )
    rejected_by_keywords = sum(
        1 for r in final_results if r.method == "adaptive_keyword_reject"
    )
    accepted_by_llm = sum(
        1 for r in final_results if r.llm_used and r.relevant
    )
    rejected_by_llm = sum(
        1 for r in final_results if r.llm_used and not r.relevant
    )
    errors = sum(1 for r in final_results if r.error)

    print(
        f"[FILTER] Готово: релевантных {len(auto_df)}/{len(df)}, "
        f"LLM-вызовов {llm_calls}/{len(df)}, "
        f"принято правилами {accepted_by_keywords}, отклонено правилами {rejected_by_keywords}, "
        f"принято LLM {accepted_by_llm}, отклонено LLM {rejected_by_llm}, "
        f"ошибок {errors}. Сохранено: {output_csv}"
    )
