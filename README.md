# Нефть Brent и курс рубля (2014–2024, месячные данные)

Установка:  pip install -r requirements.txt

Порядок запуска (из папки проекта):
1. python 01_prepare_data.py   — загрузка и подготовка данных -> data/processed/monthly.csv
2. python 02_descriptive.py    — описательная статистика, 5 рисунков
3. python 03_stationarity.py   — ADF, KPSS, Zivot-Andrews, коинтеграция, рекомендация по модели

Параметры выборки, даты событий и подвыборки — в config.py.
Если автоматическая загрузка не сработала, скрипт 01 напечатает, какие файлы положить в data/raw/.
Необязательные ряды (Urals, ключевая ставка, инфляция): data/raw/urals.csv, keyrate.csv, cpi.csv
(два столбца: дата, значение).
