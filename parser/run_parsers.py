import os

import pandas as pd

from .parsers import get_all_news
from .sources import RSS_URLS, TG_CHANNELS
from config import PARSER_HOURS, TG_API_ID, TG_API_HASH, PARSED_NEWS_CSV


def run_parser(output_csv, hours=None):
    if hours is None:
        hours = PARSER_HOURS

    tg_api_id = TG_API_ID
    tg_api_hash = TG_API_HASH

    if tg_api_id and tg_api_hash:
        try:
            tg_api_id = int(tg_api_id)
        except ValueError as e:
            raise ValueError("TG_API_ID должен быть числом") from e
    else:
        print("[PARSER] Telegram пропущен: TG_API_ID/TG_API_HASH не заданы")
        tg_api_id = None
        tg_api_hash = None

    news = get_all_news(
        rss_links=RSS_URLS,
        tg_channels=TG_CHANNELS if tg_api_id and tg_api_hash else [],
        api_id=tg_api_id,
        api_hash=tg_api_hash,
        hours=hours,
    )

    print(f"[PARSER] Собрано новостей: {len(news)}")

    df = pd.DataFrame(news, columns=["source_type", "source", "url", "date", "title", "text"])

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"[PARSER] Сохранено: {output_csv}")


if __name__ == "__main__":
    run_parser(PARSED_NEWS_CSV, hours=PARSER_HOURS)
