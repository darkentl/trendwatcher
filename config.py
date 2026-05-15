"""Настройки проекта и пути к рабочим файлам."""
import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, ".env"))


def _path_from_project(value):
    if os.path.isabs(value):
        return value
    return os.path.join(BASE_DIR, value)


# Пути к данным
DATA_DIR = _path_from_project(os.getenv("DATA_DIR", "data"))
PARSED_NEWS_CSV = os.path.join(DATA_DIR, "parsed_news.csv")
CLEANED_CSV = os.path.join(DATA_DIR, "cleaned_df.csv")
DEDUPLICATED_CSV = os.path.join(DATA_DIR, "deduplicated_primary.csv")
NEW_INFOPVODS_CSV = os.path.join(DATA_DIR, "new_infopovods.csv")
AUTO_ONLY_CSV = os.path.join(DATA_DIR, "auto_only.csv")
TRENDS_JSON = os.path.join(DATA_DIR, "trends.json")
SEEN_INFOPVODS_JSON = os.path.join(DATA_DIR, "seen_infopovods.json")

# Парсинг
PARSER_HOURS = int(os.getenv("PARSER_HOURS", "1"))
RSS_TIMEOUT = int(os.getenv("RSS_TIMEOUT", "20"))
RSS_USER_AGENT = os.getenv("RSS_USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 TrendWatcher/1.0")
RSS_FALLBACK_WITHOUT_SSL_VERIFY = os.getenv("RSS_FALLBACK_WITHOUT_SSL_VERIFY", "1") == "1"
TG_API_ID = os.getenv("TG_API_ID")
TG_API_HASH = os.getenv("TG_API_HASH")

# LLM / OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "TrendWatcher")

FILTER_MODEL = os.getenv("FILTER_MODEL", "openai/gpt-4o-mini")
CARD_MODEL = os.getenv("CARD_MODEL", "anthropic/claude-sonnet-4-5")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_RETRY_BASE_SECONDS = float(os.getenv("LLM_RETRY_BASE_SECONDS", "1"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "45"))
LLM_USER_TEXT_LIMIT = int(os.getenv("LLM_USER_TEXT_LIMIT", "6000"))

# LLM-фильтр
# По умолчанию LLM-фильтр отключен, чтобы не платить за проверку каждого инфоповода после dedup.
# В этом режиме работает локальный keyword/scoring-фильтр.
USE_LLM_FILTER = os.getenv("USE_LLM_FILTER", "0") == "1"
FILTER_UNKNOWN_WITH_LLM = os.getenv("FILTER_UNKNOWN_WITH_LLM", "0") == "1"
LLM_FILTER_KEYWORD_PREPASS = os.getenv("LLM_FILTER_KEYWORD_PREPASS", "1") == "1"
LLM_FILTER_KEYWORD_MIN_SCORE = int(os.getenv("LLM_FILTER_KEYWORD_MIN_SCORE", "2"))

# Параллельная обработка
DEDUP_WORKERS = int(os.getenv("DEDUP_WORKERS", "4"))
LLM_FILTER_WORKERS = int(os.getenv("LLM_FILTER_WORKERS", "1"))
MAKE_CARD_WORKERS = int(os.getenv("MAKE_CARD_WORKERS", "1"))

# Дедубликация
# Adaptive теперь экономнее: больше пар решаются текстовыми правилами до embeddings.
# Если нужно максимально дешево, поставьте DEDUP_USE_EMBEDDINGS=0: дедуп будет работать только текстовыми правилами.
DEDUP_USE_EMBEDDINGS = os.getenv("DEDUP_USE_EMBEDDINGS", "1") == "1"
DEDUP_LEVEL_1_THRESHOLD = float(os.getenv("DEDUP_LEVEL_1_THRESHOLD", "0.88"))
DEDUP_LEVEL_2_THRESHOLD = float(os.getenv("DEDUP_LEVEL_2_THRESHOLD", "0.80"))
DEDUP_TEXT_JACCARD_THRESHOLD = float(os.getenv("DEDUP_TEXT_JACCARD_THRESHOLD", "0.20"))
DEDUP_TEXT_SEQUENCE_THRESHOLD = float(os.getenv("DEDUP_TEXT_SEQUENCE_THRESHOLD", "0.55"))
DEDUP_TITLE_SEQUENCE_THRESHOLD = float(os.getenv("DEDUP_TITLE_SEQUENCE_THRESHOLD", "0.45"))
