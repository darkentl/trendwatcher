"""Расчёт итоговой горячести карточки по LLM-оценке, источнику и свежести."""
from datetime import datetime, timezone


def source_weight(url):
    url = str(url or "").lower()

    if "cbr.ru" in url:
        return 1.0
    if any(x in url for x in [
        "autostat.ru", "motor.ru", "gazeta.ru", "kolesa.ru",
        "110km.ru", "carexpo.ru", "drom.ru"
    ]):
        return 0.7
    if "t.me" in url:
        return 0.4

    return 0.3


def freshness_weight(date_str):
    try:
        dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return 0.5

    hours_old = max((datetime.now(timezone.utc) - dt).total_seconds() / 3600, 0)

    return max(0, 24 - hours_old) / 24


def calc_hotness(card, sources_count):
    try:
        llm_raw = int(card.get("hot_score_llm", 3))
    except (ValueError, TypeError, AttributeError):
        llm_raw = 3

    llm_raw = min(max(llm_raw, 1), 5)

    sw = source_weight(card.get("url", ""))
    fw = freshness_weight(card.get("date", ""))

    try:
        sources_count = float(sources_count)
    except:
        sources_count = 1

    sc = min(max(sources_count, 1) / 4, 1)

    base_by_llm = {
        1: 20,
        2: 35,
        3: 48,
        4: 72,
        5: 90,
    }

    score = base_by_llm[llm_raw]

    # усилители
    score += sc * 5
    score += sw * 5
    score += fw * 3

    # буст для регуляторики / ЦБ
    topic = card.get("topic", "")
    url = str(card.get("url", "")).lower()

    if topic == "регуляторика":
        score += 5

    if "cbr.ru" in url:
        score += 5

    return round(min(score, 100), 1)
