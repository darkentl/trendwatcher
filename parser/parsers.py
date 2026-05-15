"""Парсинг новостей из RSS-лент и Telegram-каналов."""
import asyncio
from datetime import datetime, timedelta, timezone
import os
import ssl
from urllib.request import Request, urlopen

import feedparser
import requests
from requests.exceptions import SSLError, RequestException
from telethon import TelegramClient
import urllib3

from config import RSS_FALLBACK_WITHOUT_SSL_VERIFY, RSS_TIMEOUT, RSS_USER_AGENT


SESSION_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sess")
RSS_HEADERS = {"User-Agent": RSS_USER_AGENT}

if RSS_FALLBACK_WITHOUT_SSL_VERIFY:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _is_ssl_certificate_error(error):
    """Проверяет, связана ли ошибка с SSL-сертификатом."""
    text = str(error).lower()
    markers = [
        "certificate_verify_failed",
        "certificate verify failed",
        "self-signed certificate",
        "unable to get local issuer certificate",
    ]
    return any(marker in text for marker in markers)


def _load_rss_with_requests(site, verify_ssl=True):
    response = requests.get(
        site,
        headers=RSS_HEADERS,
        timeout=RSS_TIMEOUT,
        verify=verify_ssl,
    )
    response.raise_for_status()
    return response.content


def _load_rss_with_urllib(site, verify_ssl=True):
    request = Request(site, headers=RSS_HEADERS)

    context = None
    if site.startswith("https://") and not verify_ssl:
        context = ssl._create_unverified_context()

    with urlopen(request, timeout=RSS_TIMEOUT, context=context) as response:
        return response.read()


def _load_rss(site):
    try:
        return _load_rss_with_requests(site, verify_ssl=True)
    except SSLError as e:
        if not RSS_FALLBACK_WITHOUT_SSL_VERIFY:
            raise

        print(f"[PARSER] RSS SSL: {site}; пробую без SSL verify")
        return _load_rss_with_requests(site, verify_ssl=False)
    except RequestException as e:
        if not _is_ssl_certificate_error(e):
            print(f"[PARSER] RSS requests: {site}; {type(e).__name__}; пробую urllib")

        try:
            return _load_rss_with_urllib(site, verify_ssl=True)
        except Exception as urllib_error:
            if RSS_FALLBACK_WITHOUT_SSL_VERIFY and _is_ssl_certificate_error(urllib_error):
                print(f"[PARSER] RSS SSL: {site}; пробую urllib без SSL verify")
                return _load_rss_with_urllib(site, verify_ssl=False)
            raise urllib_error
    except Exception as e:
        try:
            return _load_rss_with_urllib(site, verify_ssl=True)
        except Exception as urllib_error:
            if RSS_FALLBACK_WITHOUT_SSL_VERIFY and _is_ssl_certificate_error(urllib_error):
                print(f"[PARSER] RSS SSL: {site}; пробую urllib без SSL verify")
                return _load_rss_with_urllib(site, verify_ssl=False)
            raise e


def _parse_rss(site):
    raw_data = _load_rss(site)
    rss_data = feedparser.parse(raw_data)

    if getattr(rss_data, "bozo", False):
        bozo_exception = getattr(rss_data, "bozo_exception", None)
        print(f"[PARSER] RSS parse warning: {site}; {bozo_exception}")

    return rss_data


def _entry_date(item):
    post_time = getattr(item, "published_parsed", None)

    if post_time is None:
        post_time = getattr(item, "updated_parsed", None)

    if post_time is None:
        return None

    try:
        return datetime(*post_time[:6], tzinfo=timezone.utc)
    except Exception:
        return None


def get_rss_news(site_links, hours=1):
    news_list = []

    time_limit = datetime.now(timezone.utc) - timedelta(hours=hours)

    for site in site_links:
        print(f"[PARSER] RSS: {site}")

        try:
            rss_data = _parse_rss(site)
        except Exception as e:
            print(f"[PARSER] RSS ошибка: {site}; {type(e).__name__}: {e}")
            continue

        entries = getattr(rss_data, "entries", [])

        if not entries:
            print(f"[PARSER] RSS пусто: {site}")
            continue

        added_count = 0

        for item in entries:
            post_date = _entry_date(item)

            if post_date is None:
                continue

            if post_date < time_limit:
                continue

            post_title = getattr(item, "title", "") or ""
            post_summary = getattr(item, "summary", "") or ""
            post_link = getattr(item, "link", "") or ""

            full_text = post_title + " " + post_summary

            if len(full_text.strip()) < 20:
                continue

            news_list.append({
                "source_type": "rss",
                "source": site,
                "url": post_link,
                "date": post_date,
                "title": post_title,
                "text": full_text,
            })
            added_count += 1

        print(f"[PARSER] RSS готово: {site}; добавлено {added_count}")

    return news_list


async def get_tg_news_async(tg_channels, api_id, api_hash, hours=1):
    news_list = []

    time_limit = datetime.now(timezone.utc) - timedelta(hours=hours)

    try:
        client_context = TelegramClient(SESSION_PATH, api_id, api_hash)
        async with client_context as client:
            for channel in tg_channels:
                print(f"[PARSER] Telegram: {channel}")

                try:
                    messages = client.iter_messages(channel, limit=100)
                except Exception as e:
                    print(f"[PARSER] Telegram ошибка: {channel}; {type(e).__name__}: {e}")
                    continue

                try:
                    async for message in messages:
                        if message.date is None:
                            continue

                        message_date = message.date.astimezone(timezone.utc)

                        if message_date < time_limit:
                            break

                        message_text = message.text or ""

                        if len(message_text.strip()) < 20:
                            continue

                        channel_name = channel.replace("@", "")

                        news_list.append({
                            "source_type": "telegram",
                            "source": channel,
                            "url": f"https://t.me/{channel_name}/{message.id}",
                            "date": message_date,
                            "title": "",
                            "text": message_text,
                        })
                except Exception as e:
                    print(f"[PARSER] Telegram ошибка: {channel}; {type(e).__name__}: {e}")
                    continue
    except Exception as e:
        print(f"[PARSER] Telegram отключён: {type(e).__name__}: {e}")

    return news_list


def get_tg_news(tg_channels, api_id, api_hash, hours=1):
    try:
        return asyncio.run(
            get_tg_news_async(tg_channels, api_id, api_hash, hours)
        )
    except Exception as e:
        print(f"[PARSER] Telegram отключён: {type(e).__name__}: {e}")
        return []


def get_all_news(rss_links, tg_channels, api_id, api_hash, hours=1):
    rss_res = get_rss_news(rss_links, hours)

    if tg_channels and api_id and api_hash:
        tg_res = get_tg_news(tg_channels, api_id, api_hash, hours)
    else:
        tg_res = []

    all_news = rss_res + tg_res

    print(f"[PARSER] Всего собрано: {len(all_news)}")

    return all_news
