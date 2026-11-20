"""
Этапы 3–4. Загрузка, проверка и подготовка месячных данных.

Что делает скрипт:
  1. Скачивает Brent (FRED, MCOILBRENTEU, месячное среднее) и официальный
     курс USD/RUB (ЦБ РФ, ежедневный) и кэширует в data/raw/.
  2. Если сайт недоступен, использует файлы, положенные вручную
     (см. инструкцию, которую скрипт напечатает).
  3. Приводит всё к месячной частоте (среднее за месяц; конец месяца — для
     проверки устойчивости).
  4. Считает логарифмы, приросты в процентах, положительную/отрицательную
     части приростов Brent (для асимметрии) и дамми на события.
  5. Проверяет качество данных и сохраняет data/processed/monthly.csv.

Необязательные переменные подхватываются автоматически, если файл есть:
  data/raw/urals.csv, keyrate.csv, cpi.csv  (два столбца: дата, значение)
  Индекс доллара скачивается с FRED (DTWEXBGS, широкий индекс; это не ICE DXY).

Запуск:  python 01_prepare_data.py            (использовать кэш, если есть)
         python 01_prepare_data.py --refresh  (скачать заново)
"""
import argparse
import io
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import requests

import config as C

UA = {"User-Agent": "Mozilla/5.0 (compatible; research-script)"}
BRENT_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MCOILBRENTEU"
USDIDX_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS"
CBR_URL = ("https://www.cbr.ru/scripts/XML_dynamic.asp"
           "?date_req1={d1}&date_req2={d2}&VAL_NM_RQ=R01235")  # R01235 = USD

BRENT_FILE = C.RAW / "brent_fred.csv"
USDRUB_FILE = C.RAW / "usdrub_cbr.csv"
USDIDX_FILE = C.RAW / "usd_index_fred.csv"

BUFFER_START = pd.Timestamp(C.START) - pd.DateOffset(months=1)


# --------------------------------------------------------------------------
# Чтение и скачивание
# --------------------------------------------------------------------------
def parse_dates(col):
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%Y-%m"):
        d = pd.to_datetime(col, format=fmt, errors="coerce")
        if d.notna().mean() > 0.9:
            return d
    raise ValueError("не удалось распознать формат дат в первом столбце")


def read_series(path, name):
    """Читает файл из двух столбцов (дата, значение) -> pd.Series.
    Понимает разделители , ; табуляция, десятичную запятую, точку как пропуск."""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    first = text.splitlines()[0]
    sep = ";" if ";" in first else ("\t" if "\t" in first else ",")
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str)
    dates = parse_dates(df.iloc[:, 0].str.strip())
    vals = pd.to_numeric(
        df.iloc[:, 1].str.strip().str.replace(",", ".", regex=False),
        errors="coerce")
    s = pd.Series(vals.values, index=dates, name=name)
    s = s[s.index.notna()].dropna()
    return s.groupby(level=0).mean().sort_index()


def fetch_csv(url, cache, label, refresh):
    """Скачивает CSV в кэш. Возвращает True, если файл в итоге есть."""
    if cache.exists() and not refresh:
        print(f"[{label}] используется локальный файл {cache.name}")
        return True
    try:
        r = requests.get(url, headers=UA, timeout=60)
        r.raise_for_status()
        cache.write_bytes(r.content)
        print(f"[{label}] скачано -> {cache.name}")
        return True
    except Exception as e:  # сеть, блокировка, формат
        print(f"[{label}] не удалось скачать: {e}")
        return cache.exists()


def fetch_cbr(refresh):
    if USDRUB_FILE.exists() and not refresh:
        print(f"[USD/RUB] используется локальный файл {USDRUB_FILE.name}")
        return True
    d1 = BUFFER_START.strftime("%d/%m/%Y")
    d2 = pd.Timestamp(C.END).strftime("%d/%m/%Y")
    try:
        r = requests.get(CBR_URL.format(d1=d1, d2=d2), headers=UA, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.content)   # кодировка windows-1251 берётся из XML
        rows = []
        for rec in root.findall("Record"):
            d = pd.to_datetime(rec.attrib["Date"], format="%d.%m.%Y")
            nominal = float(rec.find("Nominal").text.replace(",", "."))
            value = float(rec.find("Value").text.replace(",", "."))
            rows.append((d.strftime("%Y-%m-%d"), value / nominal))
        if not rows:
            raise ValueError("ЦБ вернул пустой ответ")
        pd.DataFrame(rows, columns=["date", "usdrub"]).to_csv(
            USDRUB_FILE, index=False)
        print(f"[USD/RUB] скачано {len(rows)} дневных значений -> {USDRUB_FILE.name}")
        return True
    except Exception as e:
        print(f"[USD/RUB] не удалось скачать: {e}")
        return USDRUB_FILE.exists()


def to_month(s):
    """Среднее за месяц, индекс — первый день месяца."""
    return s.resample("MS").mean()


MANUAL_HELP = """
Не удалось получить данные автоматически. Положите файлы вручную в data/raw/:
  brent_fred.csv   — FRED, серия MCOILBRENTEU (кнопка Download -> CSV)
                     https://fred.stlouisfed.org/series/MCOILBRENTEU
  usdrub_cbr.csv   — ЦБ РФ, динамика официального курса доллара, два столбца:
                     дата;значение  (cbr.ru -> Статистика -> Курсы валют)
Затем запустите скрипт снова. Если проблема в сети, проверьте доступ к
fred.stlouisfed.org и cbr.ru (или используйте VPN).
"""


# --------------------------------------------------------------------------
# Основная сборка
# --------------------------------------------------------------------------
def build(refresh):
    ok_b = fetch_csv(BRENT_URL, BRENT_FILE, "Brent", refresh)
    ok_f = fetch_cbr(refresh)
    if not (ok_b and ok_f):
        print(MANUAL_HELP)
        sys.exit(1)

    brent = to_month(read_series(BRENT_FILE, "brent"))
    fx_daily = read_series(USDRUB_FILE, "usdrub")
    usdrub = to_month(fx_daily)
    usdrub_eom = fx_daily.resample("MS").last().rename("usdrub_eom")

    df = pd.concat([brent, usdrub, usdrub_eom], axis=1)
    df = df.loc[BUFFER_START:C.END]

    # Необязательные переменные
    notes = []
    fetch_csv(USDIDX_URL, USDIDX_FILE, "Индекс доллара", refresh)
    extras = {"usd_index": USDIDX_FILE, "urals": C.RAW / "urals.csv",
              "keyrate": C.RAW / "keyrate.csv", "cpi": C.RAW / "cpi.csv"}
    for name, path in extras.items():
        if path.exists():
            try:
                df[name] = to_month(read_series(path, name)).reindex(df.index)
                notes.append(f"доп. переменная {name}: подключена "
                             f"({df[name].notna().sum()} мес.)")
            except Exception as e:
                notes.append(f"доп. переменная {name}: ошибка чтения ({e})")
        else:
            hint = (" (Urals не публикуется EIA/FRED: возьмите месячную цену "
                    "Urals из Минфина РФ или другого источника)"
                    if name == "urals" else "")
            notes.append(f"доп. переменная {name}: файла нет, пропущена{hint}")

    # Контроль: в основных переменных пропусков быть не должно
    expected = pd.date_range(BUFFER_START, C.END, freq="MS")
    missing = expected.difference(df[["brent", "usdrub"]].dropna().index)
    if len(missing):
        print("КРИТИЧНО: нет данных за месяцы:",
              ", ".join(m.strftime("%Y-%m") for m in missing))
        print("Проверьте источники (например, FRED публикует Brent с лагом).")
        sys.exit(1)

    df = df.reindex(expected)
    df.index.name = "date"

    # Логарифмы и приросты (в процентах: 100 * Δln)
    for col in ("brent", "usdrub", "usdrub_eom", "urals", "usd_index", "cpi"):
        if col in df:
            df[f"ln_{col}"] = np.log(df[col])
    df["r"] = 100 * df["ln_brent"].diff()      # прирост Brent, %
    df["g"] = 100 * df["ln_usdrub"].diff()     # прирост USD/RUB, % (+ = ослабление рубля)
    df["g_eom"] = 100 * df["ln_usdrub_eom"].diff()
    for col in ("urals", "usd_index", "cpi"):
        if col in df:
            df[f"d_{col}"] = 100 * df[f"ln_{col}"].diff()
    if "keyrate" in df:
        df["d_keyrate"] = df["keyrate"].diff()

    # Положительная и отрицательная части прироста Brent (для NARDL/асимметрии)
    df["r_pos"] = df["r"].clip(lower=0)
    df["r_neg"] = df["r"].clip(upper=0)
    df["r_pos_cum"] = df["r_pos"].fillna(0).cumsum().where(df["r"].notna())
    df["r_neg_cum"] = df["r_neg"].fillna(0).cumsum().where(df["r"].notna())

    # Дамми
    idx = df.index
    df["D_float"] = (idx >= pd.Timestamp(C.FLOAT_DATE)).astype(int)
    df["D_covid"] = ((idx >= pd.Timestamp(C.COVID_START)) &
                     (idx <= pd.Timestamp(C.COVID_END))).astype(int)
    df["D_sanc2022"] = (idx >= pd.Timestamp(C.SANCTIONS_2022)).astype(int)

    # Убираем декабрь 2013 (использовался только для приростов)
    df = df.loc[C.START:C.END]
    return df, notes


# --------------------------------------------------------------------------
# Контроль качества
# --------------------------------------------------------------------------
def robust_outliers(x, thr=3.5):
    x = x.dropna()
    mad = np.median(np.abs(x - np.median(x)))
    if mad == 0:
        return x.iloc[0:0]
    z = (x - np.median(x)) / (1.4826 * mad)
    return x[np.abs(z) > thr]


def quality_report(df, notes):
    L = []
    L.append("ОТЧЁТ О КАЧЕСТВЕ ДАННЫХ")
    L.append("=" * 60)
    L.append(f"Период: {df.index.min():%Y-%m} — {df.index.max():%Y-%m}; "
             f"наблюдений: {len(df)}")
    exp = len(pd.date_range(C.START, C.END, freq="MS"))
    L.append(f"Ожидалось месяцев: {exp}; "
             f"совпадает: {'да' if exp == len(df) else 'НЕТ'}")
    L.append(f"Дубликаты дат: {int(df.index.duplicated().sum())}")
    L.append(f"Пропуски brent/usdrub: "
             f"{int(df[['brent', 'usdrub']].isna().sum().sum())}")
    L.append(f"Неположительные значения: "
             f"{int((df[['brent', 'usdrub']] <= 0).sum().sum())}")
    L.append(f"Brent: {df['brent'].min():.1f}–{df['brent'].max():.1f} USD/барр. "
             f"(ожидаемо 10–200: "
             f"{'ок' if df['brent'].between(10, 200).all() else 'ПРОВЕРИТЬ'})")
    L.append(f"USD/RUB: {df['usdrub'].min():.1f}–{df['usdrub'].max():.1f} "
             f"(ожидаемо 20–150: "
             f"{'ок' if df['usdrub'].between(20, 150).all() else 'ПРОВЕРИТЬ'})")
    diff = (df["usdrub"] - df["usdrub_eom"]).abs().mean()
    L.append(f"Среднее |среднемесячный − конец месяца| по USD/RUB: {diff:.2f} руб.")
    n_pre = int((df.index < pd.Timestamp(C.FLOAT_DATE)).sum())
    L.append(f"Месяцев до свободного плавания (коридор, до {C.FLOAT_DATE[:7]}): {n_pre}")
    L.append("")
    L.append("Выбросы (робастный z > 3.5), НЕ удаляются — это реальные события:")
    for col, label in (("r", "Brent, %"), ("g", "USD/RUB, %")):
        o = robust_outliers(df[col])
        s = "; ".join(f"{d:%Y-%m} ({v:+.1f})" for d, v in o.items()) or "нет"
        L.append(f"  {label}: {s}")
    L.append("")
    L.extend(notes)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true",
                    help="скачать данные заново, игнорируя кэш")
    args = ap.parse_args()

    df, notes = build(args.refresh)
    report = quality_report(df, notes)
    print("\n" + report)
    (C.TAB / "data_quality.txt").write_text(report, encoding="utf-8")

    df.to_csv(C.PROC / "monthly.csv")
    try:
        df.to_excel(C.PROC / "monthly.xlsx")
    except ImportError:
        pass  # openpyxl не установлен — достаточно CSV
    print(f"\nСохранено: data/processed/monthly.csv ({df.shape[0]} строк, "
          f"{df.shape[1]} столбцов)")


if __name__ == "__main__":
    main()
