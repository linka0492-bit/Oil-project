"""
Этап 6. Проверка временных рядов: стационарность и коинтеграция.

Что делает скрипт:
  1. ADF и KPSS для ln(Brent), ln(USD/RUB) в уровнях (спецификации 'c' и 'ct')
     и для приростов r, g.
  2. Тест Зивота-Эндрюса (Zivot-Andrews): единичный корень при одном
     эндогенном структурном сдвиге. Показывает, не «маскирует» ли разрыв
     (2014/2020/2022) результаты ADF.
  3. Порядок интегрированности каждого ряда.
  4. Коинтеграция (Энгл-Грейнджер и Йохансен) для полной выборки и подвыборок.
  5. Рекомендацию по спецификации модели для этапа 7.

Результаты: output/tables/stationarity_tests.csv, zivot_andrews.csv,
            cointegration_tests.csv, model_recommendation.txt

Запуск:  python 03_stationarity.py   (после 01_prepare_data.py)
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, coint, kpss, zivot_andrews
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import config as C

warnings.simplefilter("ignore", InterpolationWarning)  # p-значения KPSS ограничены [0.01; 0.10]

df = pd.read_csv(C.PROC / "monthly.csv", index_col="date", parse_dates=True)
df.index.freq = "MS"
print(f"Загружено {len(df)} наблюдений: {df.index.min():%Y-%m} — {df.index.max():%Y-%m}")

LABELS = {"ln_brent": "ln Brent", "ln_usdrub": "ln USD/RUB",
          "r": "Δ ln Brent (r)", "g": "Δ ln USD/RUB (g)"}
SPECS = {"ln_brent": ["c", "ct"], "ln_usdrub": ["c", "ct"], "r": ["c"], "g": ["c"]}


def verdict(adf_p, kpss_p):
    adf_rej = adf_p < C.ALPHA      # H0 ADF: единичный корень
    kpss_rej = kpss_p < C.ALPHA    # H0 KPSS: стационарность
    if adf_rej and not kpss_rej:
        return "stationary"
    if not adf_rej and kpss_rej:
        return "non-stationary"
    if adf_rej and kpss_rej:
        return "conflict"
    return "inconclusive"


# --------------------------------------------------------------------------
# 1. ADF + KPSS
# --------------------------------------------------------------------------
rows, verdicts = [], {}
for var, specs in SPECS.items():
    x = df[var].dropna()
    for reg in specs:
        a_stat, a_p, a_lag, _, _, _ = adfuller(
            x, maxlag=C.MAX_LAG, regression=reg, autolag="AIC")
        k_stat, k_p, k_lag, _ = kpss(x, regression=reg, nlags="auto")
        v = verdict(a_p, k_p)
        verdicts[(var, reg)] = v
        rows.append({
            "Variable": LABELS[var], "Deterministics": {"c": "constant", "ct": "const+trend"}[reg],
            "N": len(x), "ADF stat": a_stat, "ADF p": a_p, "ADF lags": a_lag,
            "KPSS stat": k_stat, "KPSS p": k_p, "Verdict": v,
        })
st = pd.DataFrame(rows)
st.round(4).to_csv(C.TAB / "stationarity_tests.csv", index=False)
print("\n1. ADF (H0: единичный корень) и KPSS (H0: стационарность)")
print("   KPSS p-значения ограничены интервалом [0.01; 0.10]")
print(st.round(3).to_string(index=False))

# --------------------------------------------------------------------------
# 2. Зивот-Эндрюс (один эндогенный разрыв)
# --------------------------------------------------------------------------
za_rows = []
for var in ("ln_brent", "ln_usdrub"):
    x = df[var].dropna()
    for reg, label in (("c", "break in intercept"), ("ct", "break in intercept and trend")):
        try:
            stat, p, crit, base_lag, bp = zivot_andrews(
                x, trim=0.15, maxlag=C.MAX_LAG, regression=reg, autolag="AIC")
            za_rows.append({
                "Variable": LABELS[var], "Break type": label, "ZA stat": stat,
                "p-value": p, "5% crit": crit["5%"], "Lag": base_lag,
                "Break date": f"{x.index[bp]:%Y-%m}",
                "Reject unit root (5%)": p < C.ALPHA,
            })
        except Exception as e:
            za_rows.append({"Variable": LABELS[var], "Break type": label,
                            "ZA stat": np.nan, "p-value": np.nan, "5% crit": np.nan,
                            "Lag": np.nan, "Break date": f"error: {e}",
                            "Reject unit root (5%)": np.nan})
za = pd.DataFrame(za_rows)
za.round(4).to_csv(C.TAB / "zivot_andrews.csv", index=False)
print("\n2. Zivot-Andrews (H0: единичный корень без разрыва)")
print(za.round(3).to_string(index=False))

# --------------------------------------------------------------------------
# 3. Порядок интегрированности (основной критерий: спецификация 'c')
# --------------------------------------------------------------------------
def order(level_var, diff_var):
    lv, dv = verdicts[(level_var, "c")], verdicts[(diff_var, "c")]
    if lv == "stationary":
        return "I(0)"
    if lv == "non-stationary" and dv == "stationary":
        return "I(1)"
    return "unclear"


ord_b = order("ln_brent", "r")
ord_f = order("ln_usdrub", "g")
print(f"\n3. Порядок интегрированности: ln Brent — {ord_b}; ln USD/RUB — {ord_f}")
print("   (вердикт по спецификации с константой; со трендом см. таблицу выше)")

# --------------------------------------------------------------------------
# 4. Коинтеграция
# --------------------------------------------------------------------------
def johansen_rank(stat, crit, col=1):
    """Последовательная процедура: сколько гипотез r<=i отвергнуто (col=1 -> 5%)."""
    rank = 0
    for i in range(len(stat)):
        if stat[i] > crit[i, col]:
            rank += 1
        else:
            break
    return rank


co_rows = []
for name, (a, b) in C.SAMPLES.items():
    sub = df.loc[a:b, ["ln_usdrub", "ln_brent"]].dropna()
    if len(sub) < C.MIN_OBS_COINT:
        co_rows.append({"Sample": name, "N": len(sub),
                        "Note": f"пропущено: меньше {C.MIN_OBS_COINT} наблюдений"})
        continue
    t1, p1, _ = coint(sub["ln_usdrub"], sub["ln_brent"], trend="c", autolag="aic")
    t2, p2, _ = coint(sub["ln_brent"], sub["ln_usdrub"], trend="c", autolag="aic")
    p_var = VAR(sub).select_order(maxlags=6).selected_orders["aic"]
    k = max(int(p_var) - 1, 0)
    jo = coint_johansen(sub, det_order=0, k_ar_diff=k)
    co_rows.append({
        "Sample": name, "N": len(sub),
        "EG t (USD/RUB on Brent)": t1, "EG p (USD/RUB on Brent)": p1,
        "EG p (Brent on USD/RUB)": p2,
        "Johansen k_ar_diff": k,
        "Trace r=0": jo.lr1[0], "Trace 5% crit r=0": jo.cvt[0, 1],
        "Trace r<=1": jo.lr1[1], "Trace 5% crit r<=1": jo.cvt[1, 1],
        "Rank (trace)": johansen_rank(jo.lr1, jo.cvt),
        "Rank (max-eig)": johansen_rank(jo.lr2, jo.cvm),
        "Note": "",
    })
co = pd.DataFrame(co_rows)
co.round(4).to_csv(C.TAB / "cointegration_tests.csv", index=False)
print("\n4. Коинтеграция (Энгл-Грейнджер: H0 — нет коинтеграции; "
      "Йохансен: ранг по trace и max-eig, 5%)")
print(co.round(3).to_string(index=False))

# --------------------------------------------------------------------------
# 5. Рекомендация по спецификации
# --------------------------------------------------------------------------
L = ["РЕКОМЕНДАЦИЯ ПО СПЕЦИФИКАЦИИ МОДЕЛИ (для этапа 7)", "=" * 60,
     f"ln Brent: {ord_b}; ln USD/RUB: {ord_f}", ""]
full = co[co["Sample"] == "full"].iloc[0]
eg_ok = full.get("EG p (USD/RUB on Brent)", np.nan) < C.ALPHA
jo_ok = full.get("Rank (trace)", 0) >= 1
if ord_b == "I(1)" and ord_f == "I(1)":
    if eg_ok or jo_ok:
        L.append("Ряды I(1), есть признаки коинтеграции "
                 f"(EG: {'да' if eg_ok else 'нет'}; Йохансен: {'да' if jo_ok else 'нет'}).")
        L.append("-> Основная модель: ARDL в уровнях с ECM (bounds test) или VECM;")
        L.append("   дополнительно — регрессия в приростах как проверка.")
        L.append("   Осторожно: разрывы 2014/2020/2022 могут давать ложную коинтеграцию или")
        L.append("   скрывать её; проверьте подвыборки и Zivot-Andrews выше.")
    else:
        L.append("Ряды I(1), коинтеграции не найдено.")
        L.append("-> Основная модель: регрессия с распределёнными лагами на приростах (r, g),")
        L.append("   VAR в приростах; долгосрочное соотношение не оцениваем.")
elif "unclear" in (ord_b, ord_f):
    L.append("Порядок интегрированности неоднозначен (низкая мощность тестов на 132")
    L.append("наблюдениях либо влияние разрывов). Смотрите Zivot-Andrews и спецификацию 'ct'.")
    L.append("-> Надёжный вариант: ARDL с bounds test (допускает смесь I(0)/I(1)) и")
    L.append("   параллельно модель в приростах.")
else:
    L.append("Хотя бы один ряд I(0): коинтеграция не определена.")
    L.append("-> ARDL (допускает смесь I(0)/I(1)) либо модель в приростах.")
L.append("")
L.append("Выборка начинается в 2014 г.: первые 10 месяцев — управляемый коридор,")
L.append("поэтому устойчивость результатов стоит проверить с выборкой от 2014-11.")
text = "\n".join(L)
print("\n" + text)
(C.TAB / "model_recommendation.txt").write_text(text, encoding="utf-8")
print("\nГотово. Таблицы: output/tables/")
