"""
Этап 10 (подготовка статьи). Сборка итоговых таблиц, рисунка и сводки.

Скрипт ничего не оценивает заново: он читает таблицы из output/tables/,
созданные скриптами 02–07, и собирает из них:

  output/paper_tables.xlsx   — все таблицы статьи (по листам), коэффициенты
                               в формате «оценка*** (ст. ошибка)»;
  output/tables/paper/*.csv  — те же таблицы в CSV;
  output/key_findings.txt    — основные числа и формулировки-черновики
                               (проверьте и перепишите своими словами);
  output/figures/fig16_main_results.(png|pdf) — главный рисунок статьи:
                               скользящая оценка и оценки по режимам.

Звёздочки: * p<0.10, ** p<0.05, *** p<0.01. Если какой-то файл отсутствует,
соответствующая таблица пропускается с сообщением.

Запуск:  python 08_report_tables.py   (после 02–07)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

import config as C

C.apply_plot_style()
OUT = C.BASE / "output"
PAPER = C.TAB / "paper"
PAPER.mkdir(parents=True, exist_ok=True)

sheets = {}


# --------------------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------------------
def load(name, **kw):
    p = C.TAB / name
    if not p.exists():
        print(f"  [нет файла] {name}")
        return None
    return pd.read_csv(p, **kw)


def stars(p):
    if p is None or pd.isna(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def cell(b, se, p, d=3):
    if pd.isna(b):
        return ""
    s = f"{b:.{d}f}{stars(p)}"
    return s + (f" ({se:.{d}f})" if pd.notna(se) else "")


def pf(x, d=3):
    return "" if pd.isna(x) else f"{x:.{d}f}"


def p_from_z(b, se):
    return 2 * stats.norm.sf(abs(b / se)) if se and se > 0 else np.nan


def add(name, df, note=None):
    if df is None or len(df) == 0:
        return
    sheets[name] = (df, note)
    df.to_csv(PAPER / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"  таблица {name}: {len(df)} строк")


print("Сборка таблиц:")

# --------------------------------------------------------------------------
# T1. Описательная статистика
# --------------------------------------------------------------------------
d = load("descriptive_stats.csv", index_col=0)
if d is not None:
    t = d[["N", "Mean", "Median", "Std", "Min", "Max", "Skewness",
           "Excess kurtosis", "JB p-value"]].copy()
    t.insert(0, "Variable", t.index)
    t["N"] = t["N"].astype(int)
    for c in t.columns[2:]:
        t[c] = t[c].map(lambda x: pf(x, 3))
    add("T1_descriptive", t.reset_index(drop=True),
        "Monthly data, 2014-2024. Changes are 100*Δln. JB p-value: Jarque-Bera normality test.")

# --------------------------------------------------------------------------
# T2. Единичные корни и коинтеграция
# --------------------------------------------------------------------------
st_ = load("stationarity_tests.csv")
za = load("zivot_andrews.csv")
co = load("cointegration_tests.csv")
if st_ is not None:
    t = pd.DataFrame({
        "Variable": st_["Variable"], "Deterministics": st_["Deterministics"],
        "ADF stat (p)": [f"{a:.2f} ({p:.3f})" for a, p in zip(st_["ADF stat"], st_["ADF p"])],
        "KPSS stat (p)": [f"{a:.2f} ({p:.3f})" for a, p in zip(st_["KPSS stat"], st_["KPSS p"])],
        "Verdict": st_["Verdict"]})
    add("T2a_unit_roots", t, "ADF: H0 unit root. KPSS: H0 stationarity (p-values bounded to [0.01, 0.10]).")
if za is not None:
    t = pd.DataFrame({
        "Variable": za["Variable"], "Break type": za["Break type"],
        "ZA stat": za["ZA stat"].map(lambda x: pf(x, 2)),
        "p-value": za["p-value"].map(lambda x: pf(x, 3)),
        "Break date": za["Break date"]})
    add("T2b_zivot_andrews", t, "Zivot-Andrews unit root test with one endogenous break.")
if co is not None:
    keep = co[co["N"].notna() & co["Rank (trace)"].notna()]
    t = pd.DataFrame({
        "Sample": keep["Sample"], "N": keep["N"].astype(int),
        "Engle-Granger p (USD/RUB on Brent)": keep["EG p (USD/RUB on Brent)"].map(lambda x: pf(x, 3)),
        "Johansen rank (trace)": keep["Rank (trace)"].astype(int),
        "Johansen rank (max-eig)": keep["Rank (max-eig)"].astype(int)})
    add("T2c_cointegration", t, "5% level; no cointegration found in any sample.")

# --------------------------------------------------------------------------
# T3. Основные оценки по режимам
# --------------------------------------------------------------------------
rg = load("rob_regimes3.csv")
est = load("rob_estimators.csv")
if rg is not None:
    rows = []
    for v in ("baseline", "ex-COVID"):
        if est is not None:
            e = est[(est["Variant"] == v) & (est["P"] == 1) & (est["Sample"] == "full")
                    & (est["Estimator"] == "OLS (HAC)")]
            if len(e):
                e = e.iloc[0]
                rows.append({"Variant": v, "Period": "Full sample (Feb 2014-Dec 2024)",
                             "N": int(e["N"]),
                             "beta_0 (same month)": cell(e["beta_0"], e["se b0"], p_from_z(e["beta_0"], e["se b0"])),
                             "Sum of betas": cell(e["Sum beta"], e["se"], e["p (sum=0)"]),
                             "95% CI": f"[{e['CI95 low']:.3f}; {e['CI95 high']:.3f}]", "R2": ""})
        for _, r in rg[rg["Variant"] == v].iterrows():
            rows.append({"Variant": v, "Period": r["Regime"], "N": int(r["N"]),
                         "beta_0 (same month)": cell(r["beta_0"], r["se b0"], p_from_z(r["beta_0"], r["se b0"])),
                         "Sum of betas": cell(r["Sum beta"], r["se"], r["p (sum=0)"]),
                         "95% CI": f"[{r['CI95 low']:.3f}; {r['CI95 high']:.3f}]",
                         "R2": pf(r["R2"], 3)})
    add("T3_main_estimates", pd.DataFrame(rows),
        "Dependent variable: monthly change in USD/RUB (%). Regressors: Brent change in months t and t-1. "
        "HAC (Newey-West) standard errors in parentheses. ex-COVID drops Mar-Jun 2020 and dependent lags.")

# --------------------------------------------------------------------------
# T4. Структурные изменения
# --------------------------------------------------------------------------
ch = load("breaks_chow.csv")
if ch is not None:
    pcol = "p boot (slopes)" if "p boot (slopes)" in ch.columns else "p (slopes)"
    t = pd.DataFrame({
        "Variant": ch["Variant"], "Break date": ch["Break"],
        "N before / after": [f"{a} / {b}" for a, b in zip(ch["N pre"], ch["N post"])],
        "Sum beta before": ch["Sum beta pre"].map(lambda x: pf(x, 3)),
        "Sum beta after": ch["Sum beta post"].map(lambda x: pf(x, 3)),
        "p (slopes), asymptotic HAC": ch["p (slopes)"].map(lambda x: pf(x, 3)),
        "p (slopes), wild bootstrap": ch[pcol].map(lambda x: pf(x, 3))})
    add("T4a_chow", t, "Known break dates. Use the bootstrap p-value: asymptotic HAC tests over-reject in short regimes. "
                       "The 2014-11 regime has only 9 pre-break observations.")
qa = load("breaks_quandt_andrews.csv")
if qa is not None:
    t = pd.DataFrame({
        "Variant": qa["Variant"], "Break in": qa["Break in"], "N": qa["N"],
        "supF": qa["supF"].map(lambda x: pf(x, 2)), "Estimated break": qa["Break date"],
        "Bootstrap p": qa["Bootstrap p"].map(lambda x: pf(x, 3)),
        "5% critical value": qa["Crit 5%"].map(lambda x: pf(x, 2))})
    add("T4b_supF", t, "Quandt-Andrews supF, 15% trimming, wild fixed-regressor bootstrap (999 draws).")
bp = load("breaks_multiple.csv")
if bp is not None:
    rows = []
    for v, g in bp.groupby("Variant"):
        for crit in ("BIC", "LWZ"):
            r = g.loc[g[crit].idxmin()]
            rows.append({"Variant": v, "Criterion": crit, "Selected breaks": int(r["Breaks m"]),
                         "Regime starts": r["Regime starts"]})
    add("T4c_bai_perron", pd.DataFrame(rows), "Bai-Perron-type dynamic programming, up to 3 breaks.")
pr = load("rob_regime_pairs.csv")
if pr is not None:
    pcol = "p boot (slopes)" if "p boot (slopes)" in pr.columns else "p (slopes)"
    t = pd.DataFrame({
        "Variant": pr["Variant"], "Regimes compared": pr["Pair"],
        "Difference in sum of betas (se)": [f"{b:.3f} ({s:.3f})" for b, s in zip(pr["Diff in sum beta"], pr["se"])],
        "p equal slopes (bootstrap)": pr[pcol].map(lambda x: pf(x, 3)),
        "p equal sums (bootstrap)": pr.get("p boot (equal sums)", pr["p (equal sums)"]).map(lambda x: pf(x, 3))})
    add("T4d_regime_tests", t, "R1: Feb 2014-Jan 2017; R2: Feb 2017-Feb 2022; R3: Mar 2022-Dec 2024.")
b22 = load("rob_break2022_post2017.csv")
if b22 is not None:
    t = pd.DataFrame({
        "Variant": b22["Variant"], "N": b22["N"],
        "Sum beta before / after 2022-03": [f"{a:.3f} / {b:.3f}" for a, b in
                                            zip(b22["Sum beta before"], b22["Sum beta after"])],
        "Chow p (bootstrap)": b22["Chow 2022-03: p boot (slopes)"].map(lambda x: pf(x, 3))
        if "Chow 2022-03: p boot (slopes)" in b22 else b22["Chow 2022-03: p (slopes)"].map(lambda x: pf(x, 3)),
        "supF break": b22["supF break date"], "supF bootstrap p": b22["supF bootstrap p"].map(lambda x: pf(x, 3))})
    add("T4e_break2022", t, "Test of a 2022 break within the post-budget-rule period (Feb 2017-Dec 2024).")

# --------------------------------------------------------------------------
# T5. Асимметрия
# --------------------------------------------------------------------------
asy = load("asym_results.csv")
if asy is not None:
    a = asy[asy["N"].notna() & asy["Sum bp (rises)"].notna()]
    t = pd.DataFrame({
        "Variant": a["Variant"], "Lags p": a["p"], "Sample": a["Sample"], "N": a["N"].astype(int),
        "Sum b+ (oil rises)": [cell(b, s, p) for b, s, p in zip(a["Sum bp (rises)"], a["se bp"], a["p (bp=0)"])],
        "Sum b- (oil falls)": [cell(b, s, p) for b, s, p in zip(a["Sum bn (falls)"], a["se bn"], a["p (bn=0)"])],
        "p: sums equal": a["p (sums equal)"].map(lambda x: pf(x, 3)),
        "p: lag-by-lag equal": a["p (pairwise equal)"].map(lambda x: pf(x, 3))})
    add("T5_asymmetry", t, "b-: coefficient on negative Brent changes (r- <= 0); a negative b- means oil falls weaken the ruble. "
                           "HAC standard errors; asymptotic p-values, treat borderline results with caution.")

# --------------------------------------------------------------------------
# T6–T7. Устойчивость
# --------------------------------------------------------------------------
if est is not None:
    e = est[est["P"] == 1].copy()
    e["cell"] = [cell(b, s, p) for b, s, p in zip(e["Sum beta"], e["se"], e["p (sum=0)"])]
    for v in ("baseline", "ex-COVID"):
        pv = e[e["Variant"] == v].pivot_table(index="Sample", columns="Estimator",
                                              values="cell", aggfunc="first")
        order = [c for c in ("OLS (HAC)", "OLS winsorized 1/99 (HAC)", "Huber", "LAD (median)") if c in pv.columns]
        pv = pv[order].reset_index()
        add(f"T6_robust_estimators_{v.replace('-', '_')}", pv,
            f"Sum of betas (P=1), {v}. Huber/LAD standard errors: moving block bootstrap (499 draws, block 3).")
lg = load("rob_lags.csv")
if lg is not None:
    lg = lg.copy()
    lg["cell"] = [cell(b, s, p) for b, s, p in zip(lg["Sum beta"], lg["se"], lg["p (sum=0)"])]
    for v in ("baseline", "ex-COVID"):
        pv = lg[lg["Variant"] == v].pivot_table(index="Sample", columns="P", values="cell", aggfunc="first")
        pv.columns = [f"P={c}" for c in pv.columns]
        add(f"T7_lags_{v.replace('-', '_')}", pv.reset_index(),
            f"Sum of betas by number of lags, {v} (HAC).")
eom = load("rob_eom.csv")
if eom is not None:
    eom = eom.copy()
    eom["cell"] = [cell(b, s, p) for b, s, p in zip(eom["Sum beta"], eom["se"], eom["p (sum=0)"])]
    pv = eom[(eom["P"] == 1) & (eom["Variant"] == "baseline")].pivot_table(
        index="Sample", columns="Data", values="cell", aggfunc="first").reset_index()
    add("T7b_end_of_month", pv, "Monthly averages vs end-of-month data (P=1, baseline).")

# --------------------------------------------------------------------------
# T8. Прогнозы
# --------------------------------------------------------------------------
fc = load("rob_forecast.csv")
if fc is not None:
    f = fc[~fc["Model"].isin(["Hist. mean"])].copy()
    t = pd.DataFrame({
        "Window": f["Window"], "Model": f["Model"], "N": f["N"],
        "RMSE ratio vs hist. mean": f["RMSE ratio vs hist. mean"].map(lambda x: pf(x, 3)),
        "OOS R2": f["OOS R2 vs hist. mean"].map(lambda x: pf(x, 3)),
        "DM p": f["DM p (one-sided)"].map(lambda x: pf(x, 3)),
        "Clark-West p": f["CW p (one-sided)"].map(lambda x: pf(x, 3))})
    add("T8_forecasts", t, "Expanding-window one-step-ahead forecasts of the USD/RUB change; ratio < 1 means better "
                           "than the historical mean. One-sided p-values (H1: model more accurate).")

# --------------------------------------------------------------------------
# Excel
# --------------------------------------------------------------------------
xlsx = OUT / "paper_tables.xlsx"
with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
    for name, (df_, note) in sheets.items():
        df_.to_excel(w, sheet_name=name[:31], index=False, startrow=0)
        ws = w.sheets[name[:31]]
        for i, col in enumerate(df_.columns, start=1):
            width = max([len(str(col))] + [len(str(x)) for x in df_[col].astype(str)]) + 2
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = min(width, 48)
        if note:
            ws.cell(row=len(df_) + 3, column=1, value="Note: " + note)
print(f"\nExcel: {xlsx.relative_to(C.BASE)}")

# --------------------------------------------------------------------------
# Сводка основных чисел
# --------------------------------------------------------------------------
L = ["КЛЮЧЕВЫЕ РЕЗУЛЬТАТЫ (черновик: числа взяты из таблиц, формулировки перепишите)", "=" * 70]
dj = load("dl_joint_tests.csv")
if dj is not None:
    r = dj[dj["Model"].str.startswith("DL, p=0")]
    if len(r):
        r = r.iloc[0]
        L.append(f"1. Модель только с текущим месяцем: сумма коэффициентов {r['Cumulative effect sum(beta)']:+.3f} "
                 f"(HAC se {r['se']:.3f}, p={r['p (sum=0)']:.3f}); рост Brent на 10% связан с изменением "
                 f"USD/RUB на {10 * r['Cumulative effect sum(beta)']:+.1f}%. R2={r['R2']:.2f}.")
if rg is not None:
    for v in ("baseline", "ex-COVID"):
        s = rg[rg["Variant"] == v]
        txt = "; ".join(f"{r['Regime'][:2]}: {r['Sum beta']:+.2f} [{r['CI95 low']:+.2f}; {r['CI95 high']:+.2f}]"
                        for _, r in s.iterrows())
        L.append(f"2. Режимы ({v}), сумма коэффициентов: {txt}")
if pr is not None:
    pcol = "p boot (slopes)" if "p boot (slopes)" in pr.columns else "p (slopes)"
    for _, r in pr[pr["Variant"] == "baseline"].iterrows():
        L.append(f"3. Равенство режимов {r['Pair']} (baseline): бутстрэп-p={r[pcol]:.3f}"
                 f"{' — различие значимо' if r[pcol] < 0.05 else ' — различие не доказано'}.")
if b22 is not None:
    for _, r in b22.iterrows():
        pc = r.get("Chow 2022-03: p boot (slopes)", r.get("Chow 2022-03: p (slopes)"))
        L.append(f"4. Разрыв 2022 внутри периода после бюджетного правила ({r['Variant']}): "
                 f"Чоу бутстрэп-p={pc:.3f}, supF p={r['supF bootstrap p']:.3f}.")
if asy is not None:
    a = asy[(asy["p"] == 1) & (asy["Sample"] == "full") & asy["Sum bp (rises)"].notna()]
    for _, r in a.iterrows():
        L.append(f"5. Асимметрия ({r['Variant']}, полная выборка, p=1): b+={r['Sum bp (rises)']:+.3f}, "
                 f"b-={r['Sum bn (falls)']:+.3f}, симметрия (суммы) p={r['p (sums equal)']:.3f}.")
if fc is not None:
    o = fc[fc["Model"].str.contains("oil|Nowcast", case=False) & (fc["Window"] == "all OOS")]
    if len(o):
        best = o.loc[o["RMSE ratio vs hist. mean"].idxmin()]
        L.append(f"6. Прогнозы (вся внепробная выборка): лучшая модель с нефтью — '{best['Model']}', "
                 f"RMSE относительно среднего = {best['RMSE ratio vs hist. mean']:.3f}, "
                 f"Clark-West p={best['CW p (one-sided)']:.3f}.")
(OUT / "key_findings.txt").write_text("\n".join(L), encoding="utf-8")
print("\n" + "\n".join(L))

# --------------------------------------------------------------------------
# Главный рисунок
# --------------------------------------------------------------------------
roll = load("breaks_rolling.csv", index_col=0, parse_dates=True)
if roll is not None and rg is not None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})
    ax = axes[0]
    ax.plot(roll.index, roll["Sum beta"], color="black", lw=1.5, label="36-month rolling estimate")
    ax.fill_between(roll.index, roll["CI95 low"], roll["CI95 high"], color="grey", alpha=0.25)
    bounds = [(C.START, "2017-01-01"), (C.BUDGET_RULE, "2022-02-01"), (C.SANCTIONS_2022, C.END)]
    for (a, b), (_, r) in zip(bounds, rg[rg["Variant"] == "baseline"].iterrows()):
        ax.hlines(r["Sum beta"], pd.Timestamp(a), pd.Timestamp(b), color="black", lw=2.4, ls="--",
                  label="Regime estimate" if a == C.START else None)
        ax.fill_between([pd.Timestamp(a), pd.Timestamp(b)], r["CI95 low"], r["CI95 high"],
                        color="black", alpha=0.08)
    ax.axhline(0, color="grey", lw=0.8)
    for date in C.EVENTS:
        ax.axvline(pd.Timestamp(date), color="grey", ls=":", lw=1)
    ax.set_ylabel("Sum of Brent coefficients")
    ax.set_title("Pass-through of Brent changes to USD/RUB changes")
    ax.legend(loc="lower right", fontsize=8)
    ax2 = axes[1]
    ax2.plot(roll.index, roll["R2"], color="dimgrey", lw=1.5)
    ax2.set_ylabel("R2 (rolling)")
    ax2.set_ylim(0, 1)
    for date, label in C.EVENTS.items():
        ax2.axvline(pd.Timestamp(date), color="grey", ls=":", lw=1)
        ax2.text(pd.Timestamp(date), 0.97, " " + label, rotation=90, va="top", ha="right",
                 fontsize=7, color="dimgrey")
    ax2.xaxis.set_major_locator(mdates.YearLocator(1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    C.save_fig(fig, "fig16_main_results")
else:
    print("  главный рисунок пропущен: нужны breaks_rolling.csv и rob_regimes3.csv")

print("\nГотово.")
