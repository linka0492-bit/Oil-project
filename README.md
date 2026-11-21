# Нефть Brent и курс рубля (2014–2024, месячные данные)

Установка:  pip install -r requirements.txt

Порядок запуска (из папки проекта):
1. python 01_prepare_data.py   — загрузка и подготовка данных -> data/processed/monthly.csv
2. python 02_descriptive.py    — описательная статистика, 5 рисунков
3. python 03_stationarity.py   — ADF, KPSS, Zivot-Andrews, коинтеграция, рекомендация по модели
4. python 04_distributed_lag.py — модель с распределёнными лагами (приросты), HAC, диагностика, 3 рисунка
5. python 05_structural_breaks.py — тесты разрывов (Чоу, supF с бутстрэпом, Bai-Perron), режимы, скользящая регрессия
6. python 06_asymmetry.py        — асимметрия реакции на рост/падение нефти в приростах (HAC)

7. python 07_robustness.py        — устойчивость: режимы по бюджетному правилу, робастные оценки, лаги,
                                   данные на конец месяца, контроли, внепробные прогнозы (DM, Clark-West)

8. python 08_report_tables.py     — итоговые таблицы статьи (output/paper_tables.xlsx), key_findings.txt,
                                   главный рисунок fig16 (читает таблицы из output/tables, ничего не пересчитывает)

Скрипты 05, 06, 07 используют модуль econ_utils.py (numpy/scipy, без statsmodels).
Для 07 (раздел D) перезапустите 01: он дополнительно скачает дневной Brent и посчитает r_eom.

Параметры выборки, даты событий и подвыборки — в config.py.
Если автоматическая загрузка не сработала, скрипт 01 напечатает, какие файлы положить в data/raw/.
Необязательные ряды (Urals, ключевая ставка, инфляция): data/raw/urals.csv, keyrate.csv, cpi.csv
(два столбца: дата, значение).
