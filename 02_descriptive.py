"""
Этап 5. Первичный статистический анализ.

Таблицы (output/tables):
  descriptive_stats.csv       — описательная статистика уровней и приростов
  subsample_stats.csv         — статистика и корреляции по подвыборкам и режимам
  cross_correlations.csv      — corr(g_t, r_{t-k}), k = -12..12
  high_volatility_periods.csv — периоды повышенной волатильности курса
  largest_moves.csv           — самые большие месячные изменения

Рисунки (output/figures, PNG + PDF):
  fig1_levels, fig2_changes, fig3_scatter, fig4_crosscorr, fig5_rolling

Важно: корреляция показывает статистическую связь, но не доказывает
причинное влияние нефти на курс.

Запуск:  python 02_descriptive.py   (после 01_prepare_data.py)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

import config as C

C.apply_plot_style()

df = pd.read_csv(C.PROC / "monthly.csv", index_col="date", parse_dates=True)
N = len(df)
print(f"Загружено {N} наблюдений: {df.index.min():%Y-%m} — {df.index.max():%Y-%m}")


# --------------------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------------------
def describe(x):
    x = x.dropna()
    jb, jb_p = stats.jarque_bera(x)
    return pd.Series({
        "N": len(x), "Mean": x.mean(), "Median": x.median(), "Std": x.std(),
        "Min": x.min(), "Max": x.max(), "Skewness": stats.skew(x),
        "Excess kurtosis": stats.kurtosis(x), "JB": jb, "JB p-value": jb_p,
    })


def add_events(ax, labels=False):
    ymax = ax.get_ylim()[1]
    for date, label in C.EVENTS.items():
        ax.axvline(pd.Timestamp(date), color="grey", ls=":", lw=1)
        if labels:
            ax.text(pd.Timestamp(date), ymax, " " + label, rotation=90,
                    va="top", ha="right", fontsize=8, color="dimgrey")


def fmt_dates(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def pearson(x, y):
    m = pd.concat([x, y], axis=1).dropna()
    if len(m) < 8:
        return np.nan, np.nan, len(m)
    rho, p = stats.pearsonr(m.iloc[:, 0], m.iloc[:, 1])
    return rho, p, len(m)


# --------------------------------------------------------------------------
# 1. Описательная статистика
# --------------------------------------------------------------------------
tab = pd.DataFrame({
    "Brent, USD/bbl": describe(df["brent"]),
    "USD/RUB, RUB": describe(df["usdrub"]),
    "Brent, log change, %": describe(df["r"]),
    "USD/RUB, log change, %": describe(df["g"]),
}).T
tab.round(3).to_csv(C.TAB / "descriptive_stats.csv")
print("\n1. Описательная статистика")
print(tab.round(3).to_string())

# --------------------------------------------------------------------------
# 2. Подвыборки и режимы
# --------------------------------------------------------------------------
rows = []
periods = [(n, a, b) for n, (a, b) in C.SAMPLES.items()] + list(C.REGIMES)
for name, a, b in periods:
    sub = df.loc[a:b]
    if len(sub) < 8:
        continue
    rho, p, n = pearson(sub["g"], sub["r"])
    rho_s, p_s = stats.spearmanr(sub["g"], sub["r"], nan_policy="omit")
    rho_l, p_l, _ = pearson(sub["ln_usdrub"], sub["ln_brent"])
    slope = np.polyfit(sub["r"].dropna(), sub.loc[sub["r"].notna(), "g"], 1)[0] \
        if n >= 15 else np.nan
    rows.append({
        "Period": name, "From": a[:7], "To": b[:7], "N": n,
        "Mean r": sub["r"].mean(), "Std r": sub["r"].std(),
        "Mean g": sub["g"].mean(), "Std g": sub["g"].std(),
        "Corr(g,r) Pearson": rho, "p-value": p,
        "Corr(g,r) Spearman": rho_s, "p-value (S)": p_s,
        "OLS slope g on r": slope,
        "Corr of log levels": rho_l,
    })
sub_tab = pd.DataFrame(rows)
sub_tab.round(3).to_csv(C.TAB / "subsample_stats.csv", index=False)
print("\n2. Подвыборки (g = прирост USD/RUB, r = прирост Brent, %)")
print(sub_tab.round(3).to_string(index=False))

# --------------------------------------------------------------------------
# 3. Кросс-корреляции: corr(g_t, r_{t-k})
#    k > 0: нефть опережает курс; k < 0: курс опережает нефть
# --------------------------------------------------------------------------
rows = []
for k in range(-C.MAX_LAG, C.MAX_LAG + 1):
    rho, p, n = pearson(df["g"], df["r"].shift(k))
    rows.append({"k": k, "corr": rho, "p-value": p, "N": n,
                 "ci95": 1.96 / np.sqrt(n) if n else np.nan})
cc = pd.DataFrame(rows)
cc["significant_5pct"] = cc["p-value"] < C.ALPHA
cc.round(4).to_csv(C.TAB / "cross_correlations.csv", index=False)
sig = cc[(cc["significant_5pct"]) & (cc["k"] != 0)]
print("\n3. Кросс-корреляции g_t и r_{t-k}")
print(f"  k=0: {cc.loc[cc.k == 0, 'corr'].iloc[0]:+.3f}")
print("  значимые лаги (k != 0):",
      ", ".join(f"k={int(r.k)} ({r['corr']:+.2f})" for _, r in sig.iterrows())
      or "нет")

# --------------------------------------------------------------------------
# 4. Волатильность
# --------------------------------------------------------------------------
vol_g = df["g"].rolling(12).std()
vol_r = df["r"].rolling(12).std()
thr = vol_g.quantile(0.75)
flag = (vol_g > thr).fillna(False)
grp = (flag != flag.shift()).cumsum()
per = []
for _, chunk in flag[flag].groupby(grp[flag]):
    per.append({"From": f"{chunk.index[0]:%Y-%m}", "To": f"{chunk.index[-1]:%Y-%m}",
                "Months": len(chunk),
                "Max 12m std of g": vol_g.loc[chunk.index].max()})
hv = pd.DataFrame(per)
hv.round(3).to_csv(C.TAB / "high_volatility_periods.csv", index=False)
print(f"\n4. Периоды высокой волатильности курса (12-мес. std > {thr:.2f} п.п., верхний квартиль)")
print(hv.round(2).to_string(index=False) if len(hv) else "  нет")

moves = pd.concat([
    df["g"].abs().nlargest(10).rename("abs g").to_frame().assign(
        g=df["g"], r=df["r"], series="USD/RUB"),
    df["r"].abs().nlargest(10).rename("abs g").to_frame().assign(
        g=df["g"], r=df["r"], series="Brent"),
]).rename(columns={"abs g": "abs change"})
moves.index = moves.index.strftime("%Y-%m")
moves.round(2).to_csv(C.TAB / "largest_moves.csv")

# --------------------------------------------------------------------------
# Рисунки
# --------------------------------------------------------------------------
print("\nРисунки:")
pre = pd.Timestamp(C.REGIMES[0][2]) + pd.offsets.MonthBegin(1)

# Рис. 1. Уровни
fig, ax1 = plt.subplots(figsize=(9, 4.5))
ax1.plot(df.index, df["brent"], color="black", lw=1.6, label="Brent, USD/bbl (left)")
ax1.set_ylabel("Brent, USD/bbl")
ax2 = ax1.twinx()
ax2.spines["right"].set_visible(True)
ax2.plot(df.index, df["usdrub"], color="dimgrey", lw=1.6, ls="--",
         label="USD/RUB (right)")
ax2.set_ylabel("RUB per USD")
ax2.grid(False)
ax1.axvspan(df.index[0], pre, color="lightgrey", alpha=0.5, lw=0)
add_events(ax1, labels=True)
fmt_dates(ax1)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2,
           bbox_to_anchor=(0.5, -0.08))
ax1.set_title("Brent and USD/RUB, monthly averages (shaded: managed corridor)")
C.save_fig(fig, "fig1_levels")

# Рис. 2. Приросты
fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
axes[0].bar(df.index, df["r"], width=22, color="black")
axes[0].set_ylabel("Brent, %")
axes[0].set_title("Monthly log changes, %")
axes[1].bar(df.index, df["g"], width=22, color="dimgrey")
axes[1].set_ylabel("USD/RUB, % (+ = weaker ruble)")
for ax in axes:
    add_events(ax)
fmt_dates(axes[1])
C.save_fig(fig, "fig2_changes")

# Рис. 3. Диаграмма рассеяния по режимам
fig, ax = plt.subplots(figsize=(6.5, 5.5))
markers = ["o", "s", "^"]
shades = ["silver", "dimgrey", "black"]
for (name, a, b), mk, sh in zip(C.REGIMES, markers, shades):
    sub = df.loc[a:b, ["r", "g"]].dropna()
    if len(sub) >= 15:
        slope, icpt = np.polyfit(sub["r"], sub["g"], 1)
        xs = np.linspace(sub["r"].min(), sub["r"].max(), 20)
        ax.plot(xs, icpt + slope * xs, color=sh, lw=1.2)
        lab = f"{name}, slope={slope:.2f}"
    else:
        lab = f"{name} (n={len(sub)}, no fit)"
    ax.scatter(sub["r"], sub["g"], marker=mk, s=22, color=sh,
               edgecolor="black", linewidth=0.4, label=lab)
full = df[["r", "g"]].dropna()
b_full, a_full = np.polyfit(full["r"], full["g"], 1)
xs = np.linspace(full["r"].min(), full["r"].max(), 20)
ax.plot(xs, a_full + b_full * xs, color="black", ls="--", lw=1.4,
        label=f"Full sample, slope={b_full:.2f}")
ax.axhline(0, color="grey", lw=0.6)
ax.axvline(0, color="grey", lw=0.6)
ax.set_xlabel("Brent, monthly log change, %")
ax.set_ylabel("USD/RUB, monthly log change, %")
ax.set_title("Ruble vs oil price changes by regime")
ax.legend(fontsize=8, loc="best")
C.save_fig(fig, "fig3_scatter")

# Рис. 4. Кросс-корреляции
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(cc["k"], cc["corr"], color=["black" if s else "silver"
                                   for s in cc["significant_5pct"]])
ci = cc["ci95"].median()
ax.axhline(ci, color="grey", ls="--", lw=0.8)
ax.axhline(-ci, color="grey", ls="--", lw=0.8)
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("k  (k > 0: oil leads the exchange rate; k < 0: exchange rate leads)")
ax.set_ylabel("corr(g_t, r_{t-k})")
ax.set_title("Cross-correlations (black = significant at 5%; dashed = approx. 95% band)")
C.save_fig(fig, "fig4_crosscorr")

# Рис. 5. Скользящая корреляция и волатильность
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
roll = df["g"].rolling(24).corr(df["r"])
axes[0].plot(roll.index, roll, color="black", lw=1.5)
axes[0].axhline(0, color="grey", lw=0.6)
axes[0].set_ylabel("corr(g, r)")
axes[0].set_title("Rolling 24-month correlation of USD/RUB and Brent changes")
axes[1].plot(vol_g.index, vol_g, color="black", lw=1.5, label="USD/RUB")
axes[1].plot(vol_r.index, vol_r, color="dimgrey", lw=1.5, ls="--", label="Brent")
axes[1].set_ylabel("12-month std, p.p.")
axes[1].set_title("Rolling 12-month volatility")
axes[1].legend(loc="upper left")
for ax in axes:
    add_events(ax)
fmt_dates(axes[1])
C.save_fig(fig, "fig5_rolling")

print("\nГотово. Таблицы: output/tables/, рисунки: output/figures/")
print("Напоминание: корреляция не доказывает причинности — это задача этапов 6–8.")
