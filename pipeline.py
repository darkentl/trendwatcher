from core.clean_data import run_clean
from core.deduplicate import run_deduplicate
from core.filter_new_infopovods import filter_new_infopovods
from core.step_llm_filter import run_llm_filter
from core.step_make_card import run_make_cards
from parser.run_parsers import run_parser
from config import (
    PARSED_NEWS_CSV,
    CLEANED_CSV,
    DEDUPLICATED_CSV,
    NEW_INFOPVODS_CSV,
    AUTO_ONLY_CSV,
    TRENDS_JSON,
    PARSER_HOURS
)


def main():
    """Запускает все этапы обработки новостей."""
    print("[PIPELINE] Старт")

    print("[PIPELINE] 1/5 Парсинг")
    run_parser(PARSED_NEWS_CSV, hours=PARSER_HOURS)

    print("[PIPELINE] 2/5 Очистка")
    run_clean(PARSED_NEWS_CSV, CLEANED_CSV)

    print("[PIPELINE] 3/5 Дедубликация")
    run_deduplicate(input_csv=CLEANED_CSV, output_csv=DEDUPLICATED_CSV)

    print("[PIPELINE] 4/5 Поиск новых инфоповодов")
    df_new = filter_new_infopovods(DEDUPLICATED_CSV)
    if df_new.empty:
        print("[PIPELINE] Новых инфоповодов нет")
        return

    df_new.to_csv(NEW_INFOPVODS_CSV, index=False)
    print(f"[PIPELINE] Новых инфоповодов: {len(df_new)}")

    print("[PIPELINE] 5/5 Фильтр релевантности")
    run_llm_filter(input_csv=NEW_INFOPVODS_CSV, output_csv=AUTO_ONLY_CSV)

    print("[PIPELINE] Создание карточек")
    run_make_cards(input_csv=AUTO_ONLY_CSV, output_json=TRENDS_JSON)

    print("[PIPELINE] Готово: trends.json обновлён")


if __name__ == "__main__":
    main()
