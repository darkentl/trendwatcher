"""Создание аналитических карточек из релевантных инфоповодов."""
import json
import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .hotness import calc_hotness
from .llm import make_card
from config import MAKE_CARD_WORKERS


REQUIRED_CARD_FIELDS = ["title", "summary", "why_hot", "draft_for_team", "topic", "tone"]


def load_existing(json_path):
    if not os.path.exists(json_path):
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"[CARDS] Не удалось прочитать {json_path}: битый JSON. Будет создан новый список.")
        return []

    if not isinstance(data, list):
        print(f"[CARDS] {json_path} должен содержать список. Будет создан новый список.")
        return []

    return data


BANNED_ENTITIES = {
    "банки",
    "банк",
    "российские банки",
    "дилеры",
    "дилер",
    "автокредиты",
    "автокредитование",
    "рынок",
    "автомобили",
    "новые автомобили",
    "клиенты",
    "население",
    "потребители",
    "автодилеры"
}

TOPICS = {
    "автокредиты",
    "регуляторика",
    "дилеры",
    "рынок_авто",
    "технологии",
    "клиентское_поведение",
    "господдержка",
    "риски",
    "возможности",
    "конкуренция",
    "китай",
    "аналитика"
}

TOPIC_MAP = {
    "рынок авто": "рынок_авто",
    "рынок_авто": "рынок_авто",
    "автокредитование": "автокредиты",
    "автокредиты": "автокредиты",
    "господдержка ": "господдержка",
}


def normalize_topic(card):
    topic = str(card.get("topic", "")).strip().lower()

    card["topic"] = TOPIC_MAP.get(topic, topic)

    return card

def clean_entities(card):
    entities = card.get("entities", [])

    if not isinstance(entities, list):
        return card

    cleaned = []

    for e in entities:
        e = str(e).strip()

        if not e:
            continue

        if e.lower() in BANNED_ENTITIES:
            continue

        if e.lower() in TOPICS:
            continue

        cleaned.append(e)

    card["entities"] = list(dict.fromkeys(cleaned))

    return card


def is_valid_card(card):
    return isinstance(card, dict) and all(k in card and card[k] not in (None, "") for k in REQUIRED_CARD_FIELDS)


def run_make_cards(input_csv, output_json):
    df = pd.read_csv(input_csv)

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)

    existing_cards = load_existing(output_json)
    existing_urls = {str(c.get("url", "")) for c in existing_cards if isinstance(c, dict)}

    if df.empty:
        print("[CARDS] Нет строк для создания карточек")
        return

    required_cols = {"text", "source", "url", "date"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"В {input_csv} нет обязательных колонок: {sorted(missing)}")

    rows_to_process = [
        row for row in df.fillna("").to_dict("records")
        if str(row.get("url", "")) not in existing_urls and str(row.get("text", "")).strip()
    ]

    print(f"[CARDS] Всего строк: {len(df)}, новых к обработке: {len(rows_to_process)}")

    if not rows_to_process:
        print("[CARDS] Все карточки уже построены")
        return

    def build(row):
        try:
            card = make_card(row.get("text", ""))

            # fallback если LLM не вернул оценку
            if isinstance(card, dict) and "hot_score_llm" not in card:
                card["hot_score_llm"] = 3

            # post-processing LLM output
            card = clean_entities(card)
            card = normalize_topic(card)

            # валидация карточки
            if not is_valid_card(card):
                missing = [
                    k for k in REQUIRED_CARD_FIELDS
                    if k not in (card or {}) or not (card or {}).get(k)
                ]

                print(
                    f"[CARDS] Некорректная карточка: "
                    f"нет полей {missing}, "
                    f"url={row.get('url', '?')}"
                )
                return None

            card["source"] = row.get("source", "")
            card["url"] = row.get("url", "")
            card["date"] = row.get("date", "")

            sources_count = row.get("sources_count", 1)
            card["hotness"] = calc_hotness(card, sources_count)

            return card

        except Exception as e:
            print(
                f"[CARDS] Ошибка создания карточки "
                f"[{row.get('url', '?')}]: "
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()
            return None

    new_cards = []
    print(f"[CARDS] Запуск обработки, потоков: {MAKE_CARD_WORKERS}")
    with ThreadPoolExecutor(max_workers=MAKE_CARD_WORKERS) as ex:
        futures = {ex.submit(build, row): row for row in rows_to_process}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                new_cards.append(result)
            status = "готово" if result else "ошибка"
            print(f"[CARDS] Обработка: {i}/{len(rows_to_process)}, статус: {status}", end="\r")

    print()

    if not new_cards:
        print("[CARDS] Новых карточек нет")
        return

    all_cards = existing_cards + new_cards

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_cards, f, ensure_ascii=False, indent=2)

    print(f"[CARDS] Добавлено карточек: {len(new_cards)}")
    print(f"[CARDS] Всего в базе: {len(all_cards)}")
