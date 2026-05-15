# Крайне рекомендуете развернуть решение


## Структура проекта

```text
trendwatcher/
├── config.py                         # настройки, env-переменные и пути
├── pipeline.py                       # основной пайплайн обработки
├── requirements.txt                  # зависимости Python
├── parser/
│   ├── sources.py                    # RSS-ленты и Telegram-каналы
│   ├── parsers.py                    # логика парсинга RSS/TG
│   └── run_parsers.py                # запуск парсера
├── core/
│   ├── clean_data.py                 # очистка текстов
│   ├── deduplicate.py                # дедубликация
│   ├── filter_new_infopovods.py      # фильтр новых инфоповодов
│   ├── step_llm_filter.py            # фильтр релевантности
│   ├── step_make_card.py             # генерация карточек
│   ├── hotness.py                    # расчёт hotness
│   └── llm.py                        # работа с OpenRouter/LLM/embeddings
├── data/
│   ├── parsed_news.csv               # сырые новости после парсинга
│   ├── cleaned_df.csv                # очищенные новости
│   ├── deduplicated_primary.csv      # новости после дедубликации
│   ├── new_infopovods.csv            # новые инфоповоды
│   ├── auto_only.csv                 # релевантные авто/финансовые инфоповоды
│   ├── trends.json                   # итоговые карточки
│   └── seen_infopovods.json          # история обработанных инфоповодов
└── web/
    ├── app.py                        # Flask-приложение
    ├── templates/index.html          # HTML
    └── static/style.css              # стили
```

## Требования

- Python 3.10+
- API-ключ OpenRouter для LLM и embeddings
- Telegram API ID/API Hash — опционально, нужны только для парсинга Telegram

## Установка

```bash
cd trendwatcher
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Настройка окружения

Создайте файл `.env` в корне проекта `trendwatcher/` по примеру из .env.example:
Введи ключ от OpenRouter, Telegram API ID/API Hash (опционально)

## Запуск пайплайна

Из корня проекта:

```bash
python pipeline.py
```

Пайплайн выполнит этапы:

```text
0/5 — парсинг
1/5 — очистка текста
2/5 — дедубликация
3/5 — поиск новых инфоповодов
4/5 — фильтр релевантности
5/5 — создание карточек
```

После успешного запуска итоговые карточки будут обновлены в:

```text
data/trends.json
```

```
## Запуск веб-интерфейса

Веб-приложение читает `data/trends.json`, поэтому сначала запустите пайплайн или убедитесь, что файл уже существует.

```bash
gunicorn -w 2 -b 0.0.0.0:5000 web.app:app
```

По умолчанию приложение будет доступно по адресу:

```text
http://localhost:5000
```