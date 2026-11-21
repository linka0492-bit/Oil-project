"""
Этап 7 (часть А). Базовая модель: регрессия с распределёнными лагами на приростах.

    g_t = a + sum_{i=0..p} beta_i * r_{t-i} + e_t
    g = прирост USD/RUB, % (плюс = ослабление рубля); r = прирост Brent, %.
    Ожидаемый знак beta < 0: рост нефти -> укрепление рубля.

Три спецификации:
  DL   — чистая модель с распределёнными лагами (как в плане исследования);
  ADL  — DL + g_{t-1} (на случай автокорреляции остатков);
  DUM  — DL + дамми COVID-2020 и санкции-2022 (сдвиги среднего прироста).

Что делает скрипт:
  1. Подбор числа лагов p по AIC/BIC на ОБЩЕЙ выборке (чтобы критерии были
     сопоставимы между p).
  2. Оценка итоговых моделей с робастными (HAC, Newey-West) ошибками.
  3. Совместная значимость лагов r и накопленный эффект sum(beta_i) с тестом.
  4. Диагностика остатков: Breusch-Godfrey, Ljung-Box, Breusch-Pagan, ARCH-LM,
     Jarque-Bera, RESET, CUSUM.
  5. Та же модель по подвыборкам (первый взгляд на структурные сдвиги;
     формальные тесты — в следующем скрипте).

Таблицы: dl_ic_selection.csv, dl_coefficients.csv, dl_joint_tests.csv,
         dl_diagnostics.csv, dl_by_sample.csv
Рисунки: fig6_dl_lag_profile, fig7_dl_cumulative, fig8_dl_residuals

Запуск:  python 04_distributed_lag.py   (после 01_prepare_data.py)
"""
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import (acorr_breusch_godfrey, acorr_ljungbox,
                                          breaks_cusumolsresid, het_arch,
                                          het_breuschpagan, linear_reset)
from statsmodels.stats.stattools import jarque_bera

import config as C

C.apply_plot_style()
warnings.simplefilter("ignore", FutureWarning)

df = pd.read_csv(C.PROC / "monthly.csv", index_col="date", parse_dates=True)
print(f"Загружено {len(df)} наблюдений: {df.index.min():%Y-%m} — {df.index.max():%Y-%m}")

P_GRID = [0, 1, 3, 6, 9, 12]
P_MAX = max(P_GRID)
SPECS = ["DL", "ADL", "DUM"]
PRIMARY_SPEC = "DL"


# --------------------------------------------------------------------------
# Построение регрессоров
# --------------------------------------------------------------------------
def design(spec, p):
    cols = {f"r_l{i}": df["r"].shift(i) for i in range(p + 1)}
    if spec == "ADL":
        cols["g_l1"] = df["g"].shift(1)
    if spec == "DUM":
        cols["D_covid"] = df["D_covid"].astype(float)
        cols["D_sanc2022"] = df["D_sanc2022"].astype(float)
    X = pd.concat(cols, axis=1)
    return sm.add_constant(X, has_constant="add")


def r_cols(X):
    return [c for c in X.columns if c.startswith("r_l")]


def hac_lags(n):
    return max(int(np.floor(4 * (n / 100) ** (2 / 9))), 1)


def fit(spec, p, rows=None):
    X = design(spec, p)
    y = df["g"]
    m = pd.concat([y, X], axis=1).dropna()
    if rows is not None:
        m = m.loc[m.index.intersection(rows)]
    yy, XX = m["g"], m.drop(columns="g")
    ols = sm.OLS(yy, XX).fit()
    hac = sm.OLS(yy, XX).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags(len(yy))})
    return hac, ols, XX, yy


def scalar(x):
    return float(np.squeeze(np.asarray(x)))


def cum_effect(res, X, upto=None):
    cols = r_cols(X)
    if upto is not None:
        cols = cols[:upto + 1]
    R = np.zeros(len(X.columns))
    for c in cols:
        R[list(X.columns).index(c)] = 1.0
    t = res.t_test(R)
    eff, sd = scalar(t.effect), scalar(t.sd)
    return eff, sd, scalar(t.pvalue)


def joint_test(res, X, cols=None):
    cols = cols or r_cols(X)
    R = np.zeros((len(cols), len(X.columns)))
    for k, c in enumerate(cols):
        R[k, list(X.columns).index(c)] = 1.0
    f = res.f_test(R)
    return scalar(f.fvalue), scalar(f.pvalue)


# --------------------------------------------------------------------------
# 1. Выбор числа лагов (общая выборка для всех p)
# --------------------------------------------------------------------------
common = {}
for spec in SPECS:
    Xmax = design(spec, P_MAX)
    common[spec] = pd.concat([df["g"], Xmax], axis=1).dropna().index

rows = []
for spec in SPECS:
    for p in P_GRID:
        hac, ols, X, y = fit(spec, p, rows=common[spec])
        rows.append({"Spec": spec, "p": p, "N": int(ols.nobs), "k": X.shape[1],
                     "AIC": ols.aic, "BIC": ols.bic, "Adj R2": ols.rsquared_adj})
ic = pd.DataFrame(rows)
ic.round(3).to_csv(C.TAB / "dl_ic_selection.csv", index=False)
print("\n1. Выбор числа лагов (общая выборка; меньше — лучше)")
print(ic.round(2).to_string(index=False))

sel = {}
for spec in SPECS:
    s = ic[ic["Spec"] == spec]
    sel[spec] = {"AIC": int(s.loc[s["AIC"].idxmin(), "p"]),
                 "BIC": int(s.loc[s["BIC"].idxmin(), "p"])}
print("\nВыбранное p:", {k: v for k, v in sel.items()})

# Модели для отчёта: p по BIC, по AIC и максимальное p (профиль лагов)
report = {}
for label, p in (("BIC", sel[PRIMARY_SPEC]["BIC"]), ("AIC", sel[PRIMARY_SPEC]["AIC"]),
                 ("p=12", P_MAX)):
    report.setdefault(p, []).append(label)
models = {p: (PRIMARY_SPEC, " / ".join(lbl)) for p, lbl in report.items()}
extra = [("ADL", sel["ADL"]["AIC"], "ADL, AIC"), ("DUM", sel["DUM"]["AIC"], "DUM, AIC")]

# --------------------------------------------------------------------------
# 2–4. Оценка, тесты, диагностика
# --------------------------------------------------------------------------
coef_rows, joint_rows, diag_rows, fits = [], [], [], {}
todo = [(PRIMARY_SPEC, p, lbl) for p, (_, lbl) in models.items()]
todo += [(s, p, lbl) for s, p, lbl in extra]

for spec, p, lbl in todo:
    hac, ols, X, y = fit(spec, p)
    name = f"{spec}, p={p} [{lbl}]"
    fits[name] = (hac, ols, X, y, spec, p)
    ci = hac.conf_int()
    for c in X.columns:
        coef_rows.append({"Model": name, "Term": c, "Coef": hac.params[c],
                          "HAC se": hac.bse[c], "t": hac.tvalues[c],
                          "p-value": hac.pvalues[c],
                          "CI95 low": ci.loc[c, 0], "CI95 high": ci.loc[c, 1]})
    eff, sd, pv = cum_effect(hac, X)
    fj, pj = joint_test(hac, X)
    joint_rows.append({"Model": name, "N": int(ols.nobs), "R2": ols.rsquared,
                       "Adj R2": ols.rsquared_adj, "HAC lags": hac_lags(len(y)),
                       "Joint F (all r lags = 0)": fj, "Joint p": pj,
                       "Cumulative effect sum(beta)": eff, "se": sd,
                       "p (sum=0)": pv})

    resid = ols.resid
    def add(test, stat, pval, note=""):
        diag_rows.append({"Model": name, "Test": test, "Statistic": stat,
                          "p-value": pval, "Flag (p<0.05)": bool(pval < C.ALPHA),
                          "Note": note})
    try:
        lm, lmp, _, _ = acorr_breusch_godfrey(ols, nlags=6)
        add("Breusch-Godfrey (serial corr., 6 lags)", lm, lmp)
    except Exception as e:
        add("Breusch-Godfrey", np.nan, np.nan, str(e))
    try:
        lb = acorr_ljungbox(resid, lags=[12], return_df=True)
        add("Ljung-Box (12 lags)", lb["lb_stat"].iloc[0], lb["lb_pvalue"].iloc[0])
    except Exception as e:
        add("Ljung-Box", np.nan, np.nan, str(e))
    try:
        lm, lmp, _, _ = het_breuschpagan(resid, X)
        add("Breusch-Pagan (heteroskedasticity)", lm, lmp)
    except Exception as e:
        add("Breusch-Pagan", np.nan, np.nan, str(e))
    try:
        lm, lmp, _, _ = het_arch(resid, nlags=6)
        add("ARCH-LM (6 lags)", lm, lmp, "кластеры волатильности")
    except Exception as e:
        add("ARCH-LM", np.nan, np.nan, str(e))
    try:
        jb, jbp, _, _ = jarque_bera(resid)
        add("Jarque-Bera (normality)", jb, jbp)
    except Exception as e:
        add("Jarque-Bera", np.nan, np.nan, str(e))
    try:
        rr = linear_reset(ols, power=2, test_type="fitted", use_f=True)
        add("RESET (functional form)", scalar(rr.statistic), scalar(rr.pvalue))
    except Exception as e:
        add("RESET", np.nan, np.nan, str(e))
    try:
        sup, pc, _ = breaks_cusumolsresid(resid.values, ddof=X.shape[1])
        add("CUSUM (parameter stability)", sup, pc,
            "малое p = признак нестабильности")
    except Exception as e:
        add("CUSUM", np.nan, np.nan, str(e))

coef = pd.DataFrame(coef_rows)
joint = pd.DataFrame(joint_rows)
diag = pd.DataFrame(diag_rows)
coef.round(4).to_csv(C.TAB / "dl_coefficients.csv", index=False)
joint.round(4).to_csv(C.TAB / "dl_joint_tests.csv", index=False)
diag.round(4).to_csv(C.TAB / "dl_diagnostics.csv", index=False)

print("\n2–3. Совместная значимость лагов и накопленный эффект (HAC)")
print(joint.round(3).to_string(index=False))
print("\n4. Диагностика остатков (флаг = p < 0.05)")
print(diag[["Model", "Test", "Statistic", "p-value", "Flag (p<0.05)"]]
      .round(3).to_string(index=False))
print("\nКоэффициенты при Brent (HAC):")
print(coef[coef["Term"].str.startswith("r_l")]
      [["Model", "Term", "Coef", "HAC se", "p-value"]].round(3).to_string(index=False))

# --------------------------------------------------------------------------
# 5. Подвыборки (лаги берутся из полного ряда, делится только зависимая g_t)
# --------------------------------------------------------------------------
p_sub = sel[PRIMARY_SPEC]["AIC"] if sel[PRIMARY_SPEC]["AIC"] > 0 else 3
rows = []
for name, a, b in ([(n, a_, b_) for n, (a_, b_) in C.SAMPLES.items()] + list(C.REGIMES)):
    idx = df.loc[a:b].index
    X = design(PRIMARY_SPEC, p_sub)
    n_ok = pd.concat([df["g"], X], axis=1).dropna().index.intersection(idx)
    if len(n_ok) - X.shape[1] < 20:
        rows.append({"Sample": name, "p": p_sub, "N": len(n_ok),
                     "Note": "слишком мало наблюдений — пропущено"})
        continue
    hac, ols, Xs, ys = fit(PRIMARY_SPEC, p_sub, rows=idx)
    eff, sd, pv = cum_effect(hac, Xs)
    b0 = hac.params["r_l0"]
    rows.append({"Sample": name, "p": p_sub, "N": int(ols.nobs),
                 "beta_0 (same month)": b0, "beta_0 p": hac.pvalues["r_l0"],
                 "Cumulative sum(beta)": eff, "se": sd, "p (sum=0)": pv,
                 "R2": ols.rsquared, "Note": ""})
bys = pd.DataFrame(rows)
bys.round(4).to_csv(C.TAB / "dl_by_sample.csv", index=False)
print(f"\n5. Та же модель DL (p={p_sub}) по подвыборкам и режимам")
print(bys.round(3).to_string(index=False))

# --------------------------------------------------------------------------
# Рисунки
# --------------------------------------------------------------------------
print("\nРисунки:")
name12 = [n for n in fits if "p=12" in n and n.startswith("DL")][0]
hac, ols, X, y, spec, p = fits[name12]

# Рис. 6. Профиль лагов (коэффициенты и 95% HAC-интервалы)
cols = r_cols(X)
ci = hac.conf_int().loc[cols]
fig, ax = plt.subplots(figsize=(8, 4))
xs = np.arange(len(cols))
ax.errorbar(xs, hac.params[cols], yerr=[hac.params[cols] - ci[0], ci[1] - hac.params[cols]],
            fmt="o", color="black", capsize=3, lw=1)
ax.axhline(0, color="grey", lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels([f"{i}" for i in xs])
ax.set_xlabel("Lag of Brent change, months")
ax.set_ylabel("beta_i (pp of USD/RUB per 1 pp of Brent)")
ax.set_title("Distributed-lag coefficients, DL model with 12 lags (95% HAC CI)")
C.save_fig(fig, "fig6_dl_lag_profile")

# Рис. 7. Накопленный эффект
cum = [cum_effect(hac, X, upto=i) for i in range(len(cols))]
eff = np.array([c[0] for c in cum])
sd = np.array([c[1] for c in cum])
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(xs, eff, color="black", marker="o", ms=4)
ax.fill_between(xs, eff - 1.96 * sd, eff + 1.96 * sd, color="grey", alpha=0.3)
ax.axhline(0, color="grey", lw=0.8)
ax.set_xlabel("Horizon, months")
ax.set_ylabel("Cumulative effect on USD/RUB, pp per 1 pp")
ax.set_title("Cumulative response of USD/RUB to Brent changes (DL, p=12; 95% HAC band)")
C.save_fig(fig, "fig7_dl_cumulative")

# Рис. 8. Остатки и ACF (модель по AIC)
name_aic = [n for n in fits if n.startswith(f"{PRIMARY_SPEC},") and "AIC" in n][0]
hac_a, ols_a, X_a, y_a, _, p_a = fits[name_aic]
fig, axes = plt.subplots(2, 1, figsize=(8, 6))
axes[0].plot(ols_a.resid.index, ols_a.resid, color="black", lw=1)
axes[0].axhline(0, color="grey", lw=0.8)
axes[0].set_title(f"Residuals, {name_aic}")
axes[0].set_ylabel("pp")
plot_acf(ols_a.resid, lags=24, ax=axes[1], color="black",
         vlines_kwargs={"colors": "black"})
axes[1].set_title("Residual autocorrelation")
C.save_fig(fig, "fig8_dl_residuals")

print("\nГотово. Таблицы: output/tables/dl_*.csv")
print("Интерпретация: beta_i < 0 означает укрепление рубля при росте нефти.")
