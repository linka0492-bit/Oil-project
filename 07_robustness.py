"""
Этап 9. Проверка устойчивости и прогнозное сравнение.

Зачем: скрипт 05 нашёл главный разрыв в связи нефти и курса около
2016-05 / 2017-03 (а не в 2022 г.), что совпадает с запуском бюджетного
правила (валютные операции Минфина с февраля 2017 г.). Здесь проверяется:

  A. Три режима: до бюджетного правила / 2017-02..2022-02 / с 2022-03;
     тесты равенства режимов; разрыв 2022 г. ВНУТРИ периода после 2017-02
     (чистая проверка, без влияния 2014–2016 гг.).
  B. Робастные оценки: OLS с HAC, Хубер, медианная регрессия (LAD) с блочным
     бутстрэпом, винзоризация 1%/99% (устойчивость к выбросам).
  C. Чувствительность к числу лагов P = 0, 1, 2, 3, 6.
  D. Данные на конец месяца вместо средних (эффект усреднения Уоркинга);
     нужен столбец r_eom (скрипт 01 создаёт его, если скачался дневной Brent).
  E. Контрольные переменные (если есть в data/raw) и Urals вместо Brent.
  F. Псевдо-внепробные прогнозы g_t (расширяющееся окно) против «исторического
     среднего» и «нулевого изменения»; DM и Clark-West.

Таблицы: rob_regimes3.csv, rob_regime_pairs.csv, rob_break2022_post2017.csv,
         rob_estimators.csv, rob_lags.csv, rob_eom.csv, rob_controls.csv,
         rob_forecast.csv
Рисунки: fig14_forecast_csfe, fig15_estimators_forest

Запуск:  python 07_robustness.py   (после 01_prepare_data.py; ~1 минута)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

import config as C
import econ_utils as U

C.apply_plot_style()

N_BOOT = 499          # блочный бутстрэп (Хубер, LAD)
N_BOOT_SUPF = 999     # бутстрэп для supF
N_BOOT_WALD = 499     # бутстрэп для тестов Вальда
BLOCK = 3             # длина блока
TRIM = 0.15
OOS_START = 60        # размер начального окна для прогнозов, мес.
MIN_DOF = 20

df = pd.read_csv(C.PROC / "monthly.csv", index_col="date", parse_dates=True)
print(f"Загружено {len(df)} наблюдений: {df.index.min():%Y-%m} — {df.index.max():%Y-%m}")

SAMPLES = {
    "full": (C.START, C.END),
    "R1 (to Jan 2017)": (C.START, "2017-01-01"),
    "R2 (Feb 2017-Feb 2022)": (C.BUDGET_RULE, "2022-02-01"),
    "R3 (from Mar 2022)": (C.SANCTIONS_2022, C.END),
    "post-budget-rule (from Feb 2017)": (C.BUDGET_RULE, C.END),
}
REGIMES3 = [("R1: Feb 2014-Jan 2017", C.START, "2017-01-01"),
            ("R2: Feb 2017-Feb 2022", C.BUDGET_RULE, "2022-02-01"),
            ("R3: Mar 2022-Dec 2024", C.SANCTIONS_2022, C.END)]
VARIANTS = {"baseline": False, "ex-COVID": True}


# --------------------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------------------
def make_data(P, drop_covid=False, y="g", x="r", extra=(), winsor=False):
    yy, xx = df[y].copy(), df[x].copy()
    if winsor:
        yy = yy.clip(yy.quantile(0.01), yy.quantile(0.99))
        xx = xx.clip(xx.quantile(0.01), xx.quantile(0.99))
    X = pd.DataFrame({"const": 1.0}, index=df.index)
    for i in range(P + 1):
        X[f"r_l{i}"] = xx.shift(i)
    for c in extra:
        X[c] = df[c]
    d = pd.concat([yy.rename("g"), X], axis=1).dropna()
    if drop_covid:
        bad = pd.Series(False, index=df.index)
        for i in range(P + 1):
            bad |= df["D_covid"].shift(i).fillna(0).astype(bool)
        d = d.loc[~bad.reindex(d.index).values]
    return d


def sumvec(k, P):
    R = np.zeros(k)
    R[1:P + 2] = 1.0
    return R


def summ_ols(d, P):
    """OLS + HAC: b0 и сумма коэффициентов при r."""
    f = U.ols(d["g"].values, d.drop(columns="g").values)
    V = U.cov_hac(f)
    k = f["k"]
    sm = U.lincom(f["b"], V, sumvec(k, P), f["n"] - k)
    b0_se = float(np.sqrt(V[1, 1]))
    return {"N": f["n"], "R2": f["r2"], "beta_0": f["b"][1], "se b0": b0_se,
            "Sum beta": sm[0], "se": sm[1], "p (sum=0)": sm[2],
            "CI95 low": sm[3], "CI95 high": sm[4]}


def enough(d):
    return len(d) - (d.shape[1] - 1) >= MIN_DOF


def chow_hac(d, date):
    y, X = d["g"].values, d.drop(columns="g").values
    k = X.shape[1]
    D = (d.index >= pd.Timestamp(date)).astype(float)
    n1, n2 = int((D == 0).sum()), int((D == 1).sum())
    if min(n1, n2) < max(8, k + 3):
        return None
    Z = np.column_stack([X, X * D[:, None]])
    f = U.ols(y, Z)
    V = U.cov_hac(f)
    R_all = np.hstack([np.zeros((k, k)), np.eye(k)])
    R_sl = np.zeros((k - 1, 2 * k))
    for j in range(1, k):
        R_sl[j - 1, k + j] = 1.0
    w_a, df_a, p_a = U.wald(f["b"], V, R_all)
    w_s, df_s, p_s = U.wald(f["b"], V, R_sl)
    _, pb_s = U.wald_boot(y, Z, R_sl, B=N_BOOT_WALD)
    P = k - 2
    sv = np.zeros(k); sv[1:P + 2] = 1.0
    pre = U.lincom(f["b"], V, np.concatenate([sv, np.zeros(k)]), f["n"] - f["k"])
    post = U.lincom(f["b"], V, np.concatenate([sv, sv]), f["n"] - f["k"])
    diff = U.lincom(f["b"], V, np.concatenate([np.zeros(k), sv]), f["n"] - f["k"])
    return {"N pre": n1, "N post": n2, "Wald all": w_a, "p (all)": p_a,
            "Wald slopes": w_s, "p (slopes)": p_s, "p boot (slopes)": pb_s,
            "Sum beta pre": pre[0], "Sum beta post": post[0],
            "Change": diff[0], "se change": diff[1], "p (change)": diff[2]}


# --------------------------------------------------------------------------
# A. Три режима и разрыв 2022 внутри периода после бюджетного правила
# --------------------------------------------------------------------------
print("\nA. Режимы по бюджетному правилу (модель: g на r_t и r_{t-1}, HAC)")
P_A = 1
rows, pair_rows = [], []
for v, drop in VARIANTS.items():
    d = make_data(P_A, drop)
    parts = []
    for label, a, b in REGIMES3:
        sub = d.loc[a:b]
        if not enough(sub):
            continue
        r = summ_ols(sub, P_A)
        rows.append({"Variant": v, "Regime": label, **r})
        parts.append((label, sub))
    dd = pd.concat([p[1] for p in parts])
    k = dd.shape[1] - 1
    Xs = dd.drop(columns="g").values
    blocks = []
    for lab, sub in parts:
        D = dd.index.isin(sub.index).astype(float)
        blocks.append(Xs * D[:, None])
    Z = np.column_stack(blocks)
    f = U.ols(dd["g"].values, Z)
    V = U.cov_hac(f)
    sv = sumvec(k, P_A)
    m = len(parts)
    for i in range(m):
        for j in range(i + 1, m):
            R = np.zeros((k - 1, m * k))
            for c in range(1, k):
                R[c - 1, i * k + c] = 1.0
                R[c - 1, j * k + c] = -1.0
            w, dfw, pw = U.wald(f["b"], V, R)
            _, pb_sl = U.wald_boot(dd["g"].values, Z, R, B=N_BOOT_WALD)
            Rs = np.zeros(m * k)
            Rs[i * k:(i + 1) * k] = sv
            Rs[j * k:(j + 1) * k] = -sv
            ls = U.lincom(f["b"], V, Rs, f["n"] - f["k"])
            _, pb_sum = U.wald_boot(dd["g"].values, Z, Rs.reshape(1, -1), B=N_BOOT_WALD)
            pair_rows.append({"Variant": v, "Pair": f"{parts[i][0][:2]} vs {parts[j][0][:2]}",
                              "Wald equal slopes": w, "p (slopes)": pw, "p boot (slopes)": pb_sl,
                              "Diff in sum beta": ls[0], "se": ls[1], "p (equal sums)": ls[2],
                              "p boot (equal sums)": pb_sum})
reg3 = pd.DataFrame(rows)
pairs = pd.DataFrame(pair_rows)
reg3.round(4).to_csv(C.TAB / "rob_regimes3.csv", index=False)
pairs.round(4).to_csv(C.TAB / "rob_regime_pairs.csv", index=False)
print(reg3[["Variant", "Regime", "N", "beta_0", "Sum beta", "se", "p (sum=0)"]]
      .round(3).to_string(index=False))
print("\nПопарное равенство режимов:")
print(pairs.round(3).to_string(index=False))

print("\nРазрыв 2022 внутри периода после бюджетного правила (Feb 2017 — Dec 2024)")
rows = []
for v, drop in VARIANTS.items():
    d = make_data(P_A, drop).loc[C.BUDGET_RULE:C.END]
    ch = chow_hac(d, C.SANCTIONS_2022)
    k = d.shape[1] - 1
    sup = U.supf_boot(d["g"].values, d.drop(columns="g").values, list(range(1, k)),
                      trim=TRIM, B=N_BOOT_SUPF)
    row = {"Variant": v, "N": len(d)}
    if ch:
        row.update({"Chow 2022-03: p (slopes)": ch["p (slopes)"],
                    "Chow 2022-03: p boot (slopes)": ch["p boot (slopes)"],
                    "Sum beta before": ch["Sum beta pre"], "Sum beta after": ch["Sum beta post"],
                    "Change": ch["Change"], "p (change)": ch["p (change)"]})
    row.update({"supF (slopes)": sup["sup"], "supF break date": f"{d.index[sup['tau']]:%Y-%m}",
                "supF bootstrap p": sup["p"], "supF 5% crit": sup["crit95"]})
    rows.append(row)
b22 = pd.DataFrame(rows)
b22.round(4).to_csv(C.TAB / "rob_break2022_post2017.csv", index=False)
print(b22.round(3).to_string(index=False))

# Дополнительно: тест Чоу на дате бюджетного правила (вся выборка)
for v, drop in VARIANTS.items():
    ch = chow_hac(make_data(P_A, drop), C.BUDGET_RULE)
    if ch:
        print(f"  Чоу на {C.BUDGET_RULE[:7]} ({v}): p(slopes)={ch['p (slopes)']:.3f}, "
              f"Sum beta {ch['Sum beta pre']:+.2f} -> {ch['Sum beta post']:+.2f}")

# --------------------------------------------------------------------------
# B. Робастные оценки
# --------------------------------------------------------------------------
print(f"\nB. Робастные оценки (блочный бутстрэп B={N_BOOT}, блок={BLOCK})")
rows = []
for v, drop in VARIANTS.items():
    for P in (0, 1):
        d_all = make_data(P, drop)
        d_win = make_data(P, drop, winsor=True)
        for sname, (a, b) in SAMPLES.items():
            d = d_all.loc[a:b]
            if not enough(d):
                continue
            base = {"Variant": v, "P": P, "Sample": sname, "N": len(d)}
            r = summ_ols(d, P)
            rows.append({**base, "Estimator": "OLS (HAC)", "beta_0": r["beta_0"],
                         "se b0": r["se b0"], "Sum beta": r["Sum beta"], "se": r["se"],
                         "p (sum=0)": r["p (sum=0)"], "CI95 low": r["CI95 low"],
                         "CI95 high": r["CI95 high"]})
            dw = d_win.loc[a:b]
            if enough(dw):
                r = summ_ols(dw, P)
                rows.append({**base, "Estimator": "OLS winsorized 1/99 (HAC)",
                             "beta_0": r["beta_0"], "se b0": r["se b0"],
                             "Sum beta": r["Sum beta"], "se": r["se"],
                             "p (sum=0)": r["p (sum=0)"], "CI95 low": r["CI95 low"],
                             "CI95 high": r["CI95 high"]})
            y_, X_ = d["g"].values, d.drop(columns="g").values
            for name, est in (("Huber", U.huber), ("LAD (median)", U.lad)):
                b_hat = est(y_, X_)
                draws = U.block_boot(y_, X_, est, L=BLOCK, B=N_BOOT)
                s_hat = b_hat[1:P + 2].sum()
                s_draw = draws[:, 1:P + 2].sum(1)
                se_s = s_draw.std(ddof=1)
                se_0 = draws[:, 1].std(ddof=1)
                z = s_hat / se_s if se_s > 0 else np.nan
                rows.append({**base, "Estimator": name, "beta_0": b_hat[1], "se b0": se_0,
                             "Sum beta": s_hat, "se": se_s,
                             "p (sum=0)": 2 * stats.norm.sf(abs(z)),
                             "CI95 low": s_hat - 1.96 * se_s, "CI95 high": s_hat + 1.96 * se_s})
est_tab = pd.DataFrame(rows)
est_tab.round(4).to_csv(C.TAB / "rob_estimators.csv", index=False)
show = est_tab[(est_tab["P"] == 1) & (est_tab["Variant"] == "baseline")]
print("Сумма коэффициентов (P=1, baseline):")
print(show.pivot_table(index="Sample", columns="Estimator", values="Sum beta")
      .round(3).to_string())

# --------------------------------------------------------------------------
# C. Лаги
# --------------------------------------------------------------------------
print("\nC. Чувствительность к числу лагов (OLS, HAC)")
rows = []
for v, drop in VARIANTS.items():
    for P in (0, 1, 2, 3, 6):
        d_all = make_data(P, drop)
        for sname, (a, b) in SAMPLES.items():
            d = d_all.loc[a:b]
            if not enough(d):
                continue
            r = summ_ols(d, P)
            rows.append({"Variant": v, "P": P, "Sample": sname, "N": r["N"],
                         "beta_0": r["beta_0"], "Sum beta": r["Sum beta"],
                         "se": r["se"], "p (sum=0)": r["p (sum=0)"]})
lags = pd.DataFrame(rows)
lags.round(4).to_csv(C.TAB / "rob_lags.csv", index=False)
print(lags[lags["Variant"] == "baseline"].pivot_table(
    index="Sample", columns="P", values="Sum beta").round(3).to_string())

# --------------------------------------------------------------------------
# D. Данные на конец месяца
# --------------------------------------------------------------------------
print("\nD. Данные на конец месяца (эффект усреднения)")
if "r_eom" in df.columns and "g_eom" in df.columns:
    rows = []
    for v, drop in VARIANTS.items():
        for P in (0, 1):
            for data_lab, (yc, xc) in (("monthly average", ("g", "r")),
                                       ("end of month", ("g_eom", "r_eom"))):
                d_all = make_data(P, drop, y=yc, x=xc)
                for sname, (a, b) in SAMPLES.items():
                    d = d_all.loc[a:b]
                    if not enough(d):
                        continue
                    r = summ_ols(d, P)
                    rows.append({"Variant": v, "P": P, "Data": data_lab, "Sample": sname,
                                 "N": r["N"], "beta_0": r["beta_0"], "Sum beta": r["Sum beta"],
                                 "se": r["se"], "p (sum=0)": r["p (sum=0)"]})
    eom = pd.DataFrame(rows)
    eom.round(4).to_csv(C.TAB / "rob_eom.csv", index=False)
    print(eom[(eom["P"] == 1) & (eom["Variant"] == "baseline")].pivot_table(
        index="Sample", columns="Data", values="Sum beta").round(3).to_string())
    # корреляции на лагах ±1 для проверки эффекта Уоркинга
    for lab, (yc, xc) in (("average", ("g", "r")), ("end of month", ("g_eom", "r_eom"))):
        out = []
        for k_ in (-1, 0, 1):
            m = pd.concat([df[yc], df[xc].shift(k_)], axis=1).dropna()
            out.append(f"k={k_}: {np.corrcoef(m.iloc[:, 0], m.iloc[:, 1])[0, 1]:+.2f}")
        print(f"  corr(g_t, r_t-k), {lab}: " + "; ".join(out))
else:
    print("  столбцов r_eom/g_eom нет — перезапустите 01_prepare_data.py (нужен дневной Brent с FRED).")

# --------------------------------------------------------------------------
# E. Контроли и Urals
# --------------------------------------------------------------------------
print("\nE. Контрольные переменные и Urals")
rows = []
controls = [c for c in ("d_usd_index", "d_keyrate", "d_cpi") if c in df.columns]
if controls:
    for v, drop in VARIANTS.items():
        d0 = make_data(1, drop)
        d1 = make_data(1, drop, extra=controls)
        for label, d in (("no controls", d0), ("with " + ", ".join(controls), d1)):
            for sname in ("full", "post-budget-rule (from Feb 2017)"):
                a, b = SAMPLES[sname]
                dd = d.loc[a:b]
                if enough(dd):
                    r = summ_ols(dd, 1)
                    rows.append({"Variant": v, "Spec": label, "Sample": sname, "N": r["N"],
                                 "beta_0": r["beta_0"], "Sum beta": r["Sum beta"],
                                 "se": r["se"], "p (sum=0)": r["p (sum=0)"]})
if "d_urals" in df.columns:
    for v, drop in VARIANTS.items():
        d_all = make_data(1, drop, x="d_urals")
        for sname, (a, b) in SAMPLES.items():
            dd = d_all.loc[a:b]
            if enough(dd):
                r = summ_ols(dd, 1)
                rows.append({"Variant": v, "Spec": "Urals instead of Brent", "Sample": sname,
                             "N": r["N"], "beta_0": r["beta_0"], "Sum beta": r["Sum beta"],
                             "se": r["se"], "p (sum=0)": r["p (sum=0)"]})
ctl = pd.DataFrame(rows)
if len(ctl):
    ctl.round(4).to_csv(C.TAB / "rob_controls.csv", index=False)
    print(ctl.round(3).to_string(index=False))
else:
    print("  нет данных: положите data/raw/usd_index_fred.csv, keyrate.csv, cpi.csv, urals.csv "
          "(см. README) и перезапустите 01.")

# --------------------------------------------------------------------------
# F. Внепробные прогнозы
# --------------------------------------------------------------------------
print("\nF. Псевдо-внепробные прогнозы одного шага вперёд (расширяющееся окно)")
dF = make_data(1, False)
dF["g_l1"] = df["g"].shift(1).reindex(dF.index)
dF = dF.dropna()
MODELS = {
    "AR(1)": ["g_l1"],
    "Oil lag (r_t-1)": ["r_l1"],
    "AR(1) + oil lag": ["g_l1", "r_l1"],
    "Nowcast (r_t, same month)": ["r_l0"],
    "Nowcast + oil lag": ["r_l0", "r_l1"],
}
n = len(dF)
idxF = dF.index[OOS_START:]
y_true = dF["g"].values[OOS_START:]
fc = {"Hist. mean": [], "Zero change": []}
for m in MODELS:
    fc[m] = []
for t in range(OOS_START, n):
    tr = dF.iloc[:t]
    ytr = tr["g"].values
    fc["Hist. mean"].append(ytr.mean())
    fc["Zero change"].append(0.0)
    for m, cols in MODELS.items():
        Xtr = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in cols])
        b = U.ols_b(ytr, Xtr)
        xt = np.r_[1.0, [dF.iloc[t][c] for c in cols]]
        fc[m].append(float(xt @ b))
fc = {k_: np.array(v_) for k_, v_ in fc.items()}

covid_flag = (df["D_covid"].shift(0).fillna(0).astype(bool)
              | df["D_covid"].shift(1).fillna(0).astype(bool)).reindex(idxF).values
windows = {
    "all OOS": np.ones(len(idxF), bool),
    "all OOS ex-COVID": ~covid_flag,
    "before Mar 2022": idxF < pd.Timestamp(C.SANCTIONS_2022),
    "from Mar 2022": idxF >= pd.Timestamp(C.SANCTIONS_2022),
}
rows = []
for wname, mask in windows.items():
    if mask.sum() < 12:
        continue
    y = y_true[mask]
    fb = fc["Hist. mean"][mask]
    eb = y - fb
    mse_b = np.mean(eb ** 2)
    mse_0 = np.mean((y - fc["Zero change"][mask]) ** 2)
    for mname in ["Hist. mean", "Zero change"] + list(MODELS):
        f_ = fc[mname][mask]
        e = y - f_
        row = {"Window": wname, "Model": mname, "N": int(mask.sum()),
               "RMSE": np.sqrt(np.mean(e ** 2)), "MAE": np.mean(np.abs(e)),
               "RMSE ratio vs hist. mean": np.sqrt(np.mean(e ** 2) / mse_b),
               "RMSE ratio vs zero": np.sqrt(np.mean(e ** 2) / mse_0),
               "OOS R2 vs hist. mean": 1 - np.mean(e ** 2) / mse_b}
        if mname in MODELS:
            row["DM stat"], row["DM p (one-sided)"] = U.dm_test(eb, e)
            row["CW stat"], row["CW p (one-sided)"] = U.cw_test(y, fb, f_)
        rows.append(row)
fct = pd.DataFrame(rows)
fct.round(4).to_csv(C.TAB / "rob_forecast.csv", index=False)
print(f"  оценка с {idxF[0]:%Y-%m} по {idxF[-1]:%Y-%m}, N = {len(idxF)}")
cols = ["Window", "Model", "N", "RMSE", "RMSE ratio vs hist. mean", "OOS R2 vs hist. mean",
        "DM p (one-sided)", "CW p (one-sided)"]
print(fct[cols].round(3).to_string(index=False))

# --------------------------------------------------------------------------
# Рисунки
# --------------------------------------------------------------------------
print("\nРисунки:")
fig, ax = plt.subplots(figsize=(9, 4.2))
eb_all = y_true - fc["Hist. mean"]
for mname, style in (("Oil lag (r_t-1)", ("black", "-")),
                     ("Nowcast (r_t, same month)", ("dimgrey", "--")),
                     ("AR(1)", ("silver", ":"))):
    e = y_true - fc[mname]
    ax.plot(idxF, np.cumsum(eb_all ** 2 - e ** 2), color=style[0], ls=style[1], lw=1.8,
            label=mname)
ax.axhline(0, color="grey", lw=0.8)
for date in C.EVENTS:
    if pd.Timestamp(date) >= idxF[0]:
        ax.axvline(pd.Timestamp(date), color="grey", ls=":", lw=1)
ax.xaxis.set_major_locator(mdates.YearLocator(1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_ylabel("Cumulative SFE gain vs historical mean")
ax.set_title("Out-of-sample: cumulative squared-error gain (up = better than historical mean)")
ax.legend(loc="upper left")
C.save_fig(fig, "fig14_forecast_csfe")

# Лесной график: Sum beta по выборкам и оценщикам (P=1)
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True)
est_names = ["OLS (HAC)", "OLS winsorized 1/99 (HAC)", "Huber", "LAD (median)"]
marks = {"OLS (HAC)": "o", "OLS winsorized 1/99 (HAC)": "s", "Huber": "^", "LAD (median)": "D"}
samples = list(SAMPLES)
for ax, v in zip(axes, VARIANTS):
    s = est_tab[(est_tab["P"] == 1) & (est_tab["Variant"] == v)]
    for si, sname in enumerate(samples):
        for ei, en in enumerate(est_names):
            row = s[(s["Sample"] == sname) & (s["Estimator"] == en)]
            if row.empty:
                continue
            row = row.iloc[0]
            ypos = si + (ei - 1.5) * 0.15
            ax.errorbar(row["Sum beta"], ypos,
                        xerr=[[row["Sum beta"] - row["CI95 low"]], [row["CI95 high"] - row["Sum beta"]]],
                        fmt=marks[en], color="black", ms=4, capsize=2, lw=0.9,
                        label=en if si == 0 else None)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(range(len(samples)))
    ax.set_yticklabels(samples)
    ax.invert_yaxis()
    ax.set_title(f"{v}, P=1")
    ax.set_xlabel("Sum of beta (95% CI)")
axes[0].legend(fontsize=7, loc="lower left")
C.save_fig(fig, "fig15_estimators_forest")

print("\nГотово. Таблицы: output/tables/rob_*.csv")
