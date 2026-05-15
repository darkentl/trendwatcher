"""Общие функции для работы с OpenRouter: чат, JSON-ответы и embeddings."""
from openai import OpenAI
import json
import re

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_SITE_URL,
    OPENROUTER_APP_NAME,
    FILTER_MODEL,
    CARD_MODEL,
    EMBEDDING_MODEL,
    LLM_TIMEOUT,
    LLM_USER_TEXT_LIMIT,
    USE_LLM_FILTER,
    FILTER_UNKNOWN_WITH_LLM,
    LLM_FILTER_KEYWORD_PREPASS,
    LLM_FILTER_KEYWORD_MIN_SCORE,
)

_client = None

# создаёт OpenRouter-клиент один раз при первом LLM-запросе
def get_client(): 
    global _client
    if _client is not None:
        return _client

    api_key = OPENROUTER_API_KEY
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY не задан. Добавьте ключ в .env или переменные окружения.")

    if OpenAI is None:
        raise RuntimeError("Пакет openai не установлен. Выполните: pip install -r requirements.txt")

    default_headers = {}
    if OPENROUTER_SITE_URL:
        default_headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_APP_NAME:
        default_headers["X-Title"] = OPENROUTER_APP_NAME

    _client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=LLM_TIMEOUT,
        default_headers=default_headers or None,
    )
    return _client


FILTER_PROMPT = """
Ты финтех-трендвотчер в области автокредитования и авторынка.

Определи, относится ли текст к теме:
автокредиты, дилеры, авторынок, продажи авто, регуляторика автофинансов.

Ответ ТОЛЬКО JSON без текста и без markdown:

{"relevant": true}
или
{"relevant": false}
"""

CARD_PROMPT = """
Ты senior trendwatcher Альфа-Банка в области автокредитования.

Сделай аналитическую карточку инфоповода.

ВАЖНО:
- НЕ пересказ новости
- объясняй влияние на банки, дилеров и рынок автокредитов
- думай как аналитик
- коротко, конкретно, без канцелярита

hot_score_llm:
1 — шум
2 — слабый сигнал
3 — заметный сигнал
4 — сильный сигнал
5 — критически важно

title:
- коротко, аналитично
- до 120 символов

summary:
- 1–3 предложения
- что реально произошло

why_hot:
- почему это важно
- как повлияет на банки / дилеров / автокредиты
- что может измениться дальше
- объясняй механизм влияния на рынок
- пиши кратко, как аналитическую заметку
- НЕ используй шаблон:
  "Для банков / Для дилеров / Для рынка"
- избегай повторов и воды
- 1 аналитический абзац
- максимум 80 слов
- если >80 слов — ответ неверный
- не более 4 предложений

draft_for_team:
- сообщение в рабочий чат
- 3–5 предложений
- без "Уважаемые коллеги", "С уважением"
- заканчивай конкретным выводом или действием
- избегай фраз:
  "важно отслеживать"
  "нужно быть готовыми"
  "готовимся к"
  "следим за"

entities:
- сначала конкретные компании, бренды, регуляторы, продукты
- если их нет — предметные сущности инфоповода
- запрещены:
  банки, дилеры, клиенты, рынок, население,
  потребители, автомобили
- не используй topic как entity
- не оставляй [] если есть смысловые сущности

topic:
строго одно значение:

автокредиты
регуляторика
дилеры
рынок_авто
технологии
клиентское_поведение
господдержка
риски
конкуренция
китай
аналитика

Правила:
- ЦБ, ставка, закон, утильсбор → регуляторика
- кредиты, ставки, лизинг → автокредиты
- AI, IT, fintech → технологии
- дилерские сети → дилеры
- льготы, субсидии → господдержка
- просрочка, дефолты, fraud, падение спроса → риски
- китайские бренды → китай
- иначе → рынок_авто

ВАЖНО:
- не выдумывай вторичные эффекты без явной связи
- не спекулируй
- если влияние косвенное — так и напиши
- выводы должны быть реалистичными и проверяемыми

Ответ ТОЛЬКО валидный JSON.
Без markdown.
Без текста до/после JSON.
Используй \\n внутри строк.

{
  "title": "",
  "summary": "",
  "why_hot": "",
  "draft_for_team": "",
  "entities": [],
  "topic": "",
  "tone": "рост | падение | изменение | риск | возможность",
  "hot_score_llm": 1
}
"""

KEYWORDS = [
    "авто", "автомоб", "машин", "автокредит", "кредит", "лизинг", "дилер",
    "автодилер", "продажи", "утильсбор", "господдерж", "электромоб",
    "китай", "транспорт", "ставк", "цб", "банк", "осаго", "каско",
    "запчаст", "маркетплейс", "ключев", "инфляц", "регулятор",
    "автоваз", "lada", "vesta", "веста", "иномарк", "кроссовер",
]

STRONG_RELEVANCE_PHRASES = [
    "автокредит",
    "авто кредит",
    "льготн автокредит",
    "автомобил",
    "авторын",
    "автодилер",
    "дилер",
    "автозапчаст",
    "запчаст",
    "осаго",
    "каско",
    "утильсбор",
    "китайск",
    "электромоб",
    "автоваз",
    "lada",
    "vesta",
    "веста",
    "кроссовер",
]

MACRO_RELEVANCE_PHRASES = [
    "ключев ставк",
    "банк россии",
    "центробанк",
    "цб",
    "инфляционн риск",
    "ставк без измен",
    "ставк сохран",
]


def clean_bad_chars(s):
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1F\x7F]", "", str(s or ""))


def strip_markdown_fence(raw):
    raw = str(raw or "").strip()
    raw = re.sub(r"^```[a-z]*\s*\n?", "", raw)
    raw = re.sub(r"\n?```\s*$", "", raw)
    return raw.strip()


def escape_newlines_in_strings(raw):
    result = []
    in_string = False
    escaped = False

    for ch in str(raw or ""):
        if escaped:
            result.append(ch)
            escaped = False
            continue

        if ch == "\\" and in_string:
            result.append(ch)
            escaped = True
            continue

        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue

        if ch == "\n" and in_string:
            result.append("\\n")
        elif ch == "\r" and in_string:
            result.append("\\r")
        elif ch == "\t" and in_string:
            result.append("\\t")
        else:
            result.append(ch)

    return "".join(result)


def repair_json(raw):
    raw = strip_markdown_fence(raw)

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("JSON-объект не найден")

    raw = raw[start:end + 1]
    raw = escape_newlines_in_strings(raw)
    raw = re.sub(r",\s*}", "}", raw)
    raw = re.sub(r",\s*]", "]", raw)

    return raw


def extract_json(content):
    content = str(content or "")
    attempts = [content, strip_markdown_fence(content)]

    for raw in attempts:
        try:
            return json.loads(raw)
        except Exception:
            pass

    try:
        fixed = repair_json(content)
        return json.loads(fixed)
    except Exception as e:
        print(f"[LLM] Некорректный JSON после восстановления: {content[:600]} ...")
        raise e


def _extract_response_content(resp):
    if resp is None:
        raise ValueError("LLM вернула пустой объект ответа")

    choices = getattr(resp, "choices", None)
    if not choices:
        raise ValueError("В ответе LLM нет choices")

    first = choices[0]
    if first is None:
        raise ValueError("Пустой choice в ответе LLM")

    message = getattr(first, "message", None)
    if message is None:
        raise ValueError("В choice нет message")

    content = getattr(message, "content", None)
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict):
                chunks.append(str(part.get("text") or part.get("content") or ""))
            else:
                chunks.append(str(part or ""))
        content = "".join(chunks)

    content = clean_bad_chars(content)
    if not content.strip():
        raise ValueError("LLM вернула пустой текст")

    return content.strip()


def ask_llm(model, system, user, temp=0):
    user = str(user or "")[:LLM_USER_TEXT_LIMIT]
    model = str(model or "").strip()

    resp = get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temp,
        max_tokens=1200
    )

    content = _extract_response_content(resp)
    return content


def _compact_ru_text(text):
    text = str(text or "").lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def keyword_relevance_score(text):
    text = _compact_ru_text(text)
    score = 0
    reasons = []

    for phrase in STRONG_RELEVANCE_PHRASES:
        if phrase in text:
            score += 2
            reasons.append(phrase)

    for phrase in MACRO_RELEVANCE_PHRASES:
        if phrase in text:
            score += 2
            reasons.append(phrase)

    for keyword in KEYWORDS:
        if keyword in text:
            score += 1
            reasons.append(keyword)

    # Комбинации для смежных, но важных тем.
    # Например: маркетплейсы + автозапчасти или ЦБ + ставка.
    if "маркетплейс" in text and ("автозапчаст" in text or "запчаст" in text or "авто" in text):
        score += 4
        reasons.append("маркетплейсы+автозапчасти")

    if ("цб" in text or "банк россии" in text or "центробанк" in text) and "ставк" in text:
        score += 4
        reasons.append("цб+ставка")

    if "банк" in text and "автокредит" in text:
        score += 4
        reasons.append("банк+автокредит")

    return score, sorted(set(reasons))


def is_relevant_keyword_fallback(text):
    score, reasons = keyword_relevance_score(text)
    return score >= LLM_FILTER_KEYWORD_MIN_SCORE


def is_relevant(text):
    if not USE_LLM_FILTER and not FILTER_UNKNOWN_WITH_LLM:
        score, reasons = keyword_relevance_score(text)
        result = score >= LLM_FILTER_KEYWORD_MIN_SCORE
        return result

    if LLM_FILTER_KEYWORD_PREPASS:
        score, reasons = keyword_relevance_score(text)
        if score >= LLM_FILTER_KEYWORD_MIN_SCORE:
            return True

    if not USE_LLM_FILTER and FILTER_UNKNOWN_WITH_LLM:
        score, reasons = keyword_relevance_score(text)
        if score >= LLM_FILTER_KEYWORD_MIN_SCORE:
            return True
        if score <= 0:
            return False

    try:
        content = ask_llm(
            model=FILTER_MODEL,
            system=FILTER_PROMPT,
            user=text,
            temp=0,
        )
        data = extract_json(content)
        llm_result = bool(data.get("relevant", False))

        if not llm_result:
            fallback = is_relevant_keyword_fallback(text)
            if fallback:
                score, reasons = keyword_relevance_score(text)
                return True

        return llm_result
    except Exception as e:
        fallback = is_relevant_keyword_fallback(text)
        print(f"[LLM] Ошибка фильтра: {type(e).__name__}: {e}. Решение по ключевым словам: {fallback}")
        return fallback


def make_card(text):
    """Создаёт аналитическую карточку через LLM."""
    content = ask_llm(
        model=CARD_MODEL,
        system=CARD_PROMPT,
        user=text,
        temp=0.2,
    )
    data = extract_json(content)
    if not isinstance(data, dict):
        raise ValueError("LLM вернула JSON, но это не объект")
    return data


def get_embedding(text):
    text = str(text or "").strip()
    if not text:
        text = "empty"
    resp = get_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=text[:8000],
    )
    data = getattr(resp, "data", None)
    if not data:
        raise ValueError("В embedding-ответе нет data")
    return data[0].embedding
