"""
Этап 8, часть А. Структурные изменения связи «нефть — курс рубля».

Модель (приросты, раз коинтеграции нет — см. результаты скрипта 03):
    g_t = a + b0 * r_t + b1 * r_{t-1} + e_t        (число лагов — P_BREAK)

Методы:
  1. Тест Чоу на известных датах (ноябрь 2014, март 2020, март 2022) с
     робастной (HAC) ковариацией; отдельно — разрыв только в наклонах.
     Для малых режимов добавлены бутстрэп-p-value (асимптотические HAC-p
     слишком оптимистичны: ~15% ложных отвержений при номинальных 5%).
  2. Quandt-Andrews supF: неизвестная дата разрыва, p-value по «дикому»
     бутстрэпу при фиксированных регрессорах (999 повторов).
  3. Несколько разрывов (Bai-Perron, динамическое программирование), число
     разрывов по BIC и LWZ.
  4. Оценки по режимам (до/после 2022) и тест их равенства.
  5. Скользящая регрессия (окно 36 мес.).

Все тесты считаются в двух вариантах: «baseline» (вся выборка) и «ex-COVID»
(исключены месяцы март–июнь 2020 и следующий за ними, где лаг попадает в шок):
в эти месяцы Brent падал на 55% и возвращался на 47%, и они легко определяют
результат в одиночку.

Таблицы: breaks_chow.csv, breaks_quandt_andrews.csv, breaks_multiple.csv,
         breaks_segments.csv, breaks_regimes.csv, breaks_rolling.csv
Рисунки: fig9_break_F_path, fig10_rolling_betas, fig11_segment_betas

Запуск:  python 05_structural_breaks.py   (после 01_prepare_data.py)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

import config as C
import econ_utils as U

C.apply_plot_style()

P_BREAK = 1        # число лагов r в модели (0 = только текущий месяц)
TRIM = 0.15        # доля обрезки для supF и минимальная длина режима
N_BOOT = 999       # повторов бутстрэпа (supF)
N_BOOT_WALD = 499  # повторов бутстрэпа для тестов Вальда (Чоу, равенство режимов)
ROLL_WIN = 36      # окно скользящей регрессии, мес.
MAX_BREAKS = 3     # максимум разрывов для Bai-Perron

df = pd.read_csv(C.PROC / "monthly.csv", index_col="date", parse_dates=True)
print(f"Загружено {len(df)} наблюдений: {df.index.min():%Y-%m} — {df.index.max():%Y-%m}")

REGIME_SETS = {
    "pre-2022 vs post-2022": [("Jan 2014-Feb 2022", C.START, "2022-02-01"),
                              ("Mar 2022-Dec 2024", C.SANCTIONS_2022, C.END)],
    "float era vs post-2022": [("Nov 2014-Feb 2022", C.FLOAT_DATE, "2022-02-01"),
                               ("Mar 2022-Dec 2024", C.SANCTIONS_2022, C.END)],
}
CHOW_DATES = {"float (2014-11)": C.FLOAT_DATE, "COVID/OPEC+ (2020-03)": C.COVID_START,
              "sanctions (2022-03)": C.SANCTIONS_2022}


def make_data(P, drop_covid):
    X = pd.DataFrame({"const": 1.0}, index=df.index)
    for i in range(P + 1):
        X[f"r_l{i}"] = df["r"].shift(i)
    d = pd.concat([df["g"], X], axis=1).dropna()
    if drop_covid:
        bad = pd.Series(False, index=df.index)
        for i in range(P + 1):
            bad |= df["D_covid"].shift(i).fillna(0).astype(bool)
        d = d.loc[~bad.reindex(d.index).values]
    return d


def slope_sum_row(k):
    """Вектор для суммы коэффициентов при r (столбцы 1..k-1)."""
    R = np.zeros(k)
    R[1:] = 1.0
    return R


VARIANTS = {"baseline": False, "ex-COVID": True}
data = {v: make_data(P_BREAK, drop) for v, drop in VARIANTS.items()}
for v, d in data.items():
    print(f"  {v}: N = {len(d)}")

# --------------------------------------------------------------------------
# 1. Тест Чоу на известных датах (HAC)
# --------------------------------------------------------------------------
rows = []
for v, d in data.items():
    y = d["g"].values
    X = d.drop(columns="g").values
    k = X.shape[1]
    for label, date in CHOW_DATES.items():
        D = (d.index >= pd.Timestamp(date)).astype(float)
        n1, n2 = int((D == 0).sum()), int((D == 1).sum())
        if v == "ex-COVID" and date == C.COVID_START:
            continue
        if min(n1, n2) < max(8, k + 3):
            rows.append({"Variant": v, "Break": label, "N pre": n1, "N post": n2,
                         "Note": "слишком короткий режим — пропущено"})
            continue
        Z = np.column_stack([X, X * D[:, None]])
        f = U.ols(y, Z)
        V = U.cov_hac(f)
        R_all = np.hstack([np.zeros((k, k)), np.eye(k)])
        R_slope = np.zeros((k - 1, 2 * k))
        for j in range(1, k):
            R_slope[j - 1, k + j] = 1.0
        w_all, df_all, p_all = U.wald(f["b"], V, R_all)
        w_sl, df_sl, p_sl = U.wald(f["b"], V, R_slope)
        _, pb_all = U.wald_boot(y, Z, R_all, B=N_BOOT_WALD)
        _, pb_sl = U.wald_boot(y, Z, R_slope, B=N_BOOT_WALD)
        R_pre = np.concatenate([slope_sum_row(k), np.zeros(k)])
        R_post = np.concatenate([slope_sum_row(k), slope_sum_row(k)])
        pre = U.lincom(f["b"], V, R_pre, f["n"] - f["k"])
        post = U.lincom(f["b"], V, R_post, f["n"] - f["k"])
        diff = U.lincom(f["b"], V, np.concatenate([np.zeros(k), slope_sum_row(k)]),
                        f["n"] - f["k"])
        rows.append({"Variant": v, "Break": label, "N pre": n1, "N post": n2,
                     "Wald all coefs": w_all, "df": df_all, "p (all)": p_all,
                     "Wald slopes only": w_sl, "df slopes": df_sl, "p (slopes)": p_sl,
                     "p boot (all)": pb_all, "p boot (slopes)": pb_sl,
                     "Sum beta pre": pre[0], "se pre": pre[1],
                     "Sum beta post": post[0], "se post": post[1],
                     "Change in sum beta": diff[0], "p (change)": diff[2], "Note": ""})
chow = pd.DataFrame(rows)
chow.round(4).to_csv(C.TAB / "breaks_chow.csv", index=False)
print("\n1. Тест Чоу на известных датах (HAC; H0: нет разрыва)")
print(chow.drop(columns=["Note"], errors="ignore").round(3).to_string(index=False))

# --------------------------------------------------------------------------
# 2. Quandt-Andrews supF, неизвестная дата
# --------------------------------------------------------------------------
rows, qa_cache = [], {}
print(f"\n2. supF (неизвестная дата), бутстрэп B={N_BOOT} — это может занять до минуты")
for v, d in data.items():
    y = d["g"].values
    X = d.drop(columns="g").values
    k = X.shape[1]
    for label, idx in (("slopes only", list(range(1, k))), ("all coefficients", list(range(k)))):
        res = U.supf_boot(y, X, idx, trim=TRIM, B=N_BOOT)
        qa_cache[(v, label)] = (d.index, res)
        rows.append({"Variant": v, "Break in": label, "N": len(d),
                     "supF": res["sup"], "Break date": f"{d.index[res['tau']]:%Y-%m}",
                     "Bootstrap p": res["p"], "Crit 10%": res["crit90"],
                     "Crit 5%": res["crit95"], "Crit 1%": res["crit99"]})
qa = pd.DataFrame(rows)
qa.round(4).to_csv(C.TAB / "breaks_quandt_andrews.csv", index=False)
print(qa.round(3).to_string(index=False))

# --------------------------------------------------------------------------
# 3. Несколько разрывов (Bai-Perron)
# --------------------------------------------------------------------------
rows, seg_rows, bp_cache = [], [], {}
print("\n3. Несколько разрывов (чистая структурная смена, все коэффициенты)")
for v, d in data.items():
    y = d["g"].values
    X = d.drop(columns="g").values
    n, k = X.shape
    h = max(int(np.ceil(TRIM * n)), k + 3)
    M = min(MAX_BREAKS, max(n // h - 1, 0))
    br = U.bp_breaks(y, X, h, M)
    crit = {}
    for m, (rss, starts) in br.items():
        bic, lwz = U.bp_criteria(rss, n, k, m)
        crit[m] = (bic, lwz)
        rows.append({"Variant": v, "Breaks m": m, "RSS": rss, "BIC": bic, "LWZ": lwz,
                     "Regime starts": ", ".join(f"{d.index[s]:%Y-%m}" for s in starts) or "-"})
    m_bic = min(crit, key=lambda m: crit[m][0])
    m_lwz = min(crit, key=lambda m: crit[m][1])
    bp_cache[v] = (d, br, m_bic, m_lwz)
    for crit_name, m in (("BIC", m_bic), ("LWZ", m_lwz)):
        starts = [0] + br[m][1] + [n]
        for s_i in range(len(starts) - 1):
            a, b = starts[s_i], starts[s_i + 1]
            f = U.ols(y[a:b], X[a:b])
            V = U.cov_hac(f)
            sm = U.lincom(f["b"], V, slope_sum_row(k), f["n"] - f["k"])
            seg_rows.append({
                "Variant": v, "Selected by": crit_name, "m": m,
                "Regime": f"{d.index[a]:%Y-%m} to {d.index[b - 1]:%Y-%m}", "N": b - a,
                "beta_0": f["b"][1], "se b0": np.sqrt(V[1, 1]),
                "Sum beta": sm[0], "se": sm[1], "p (sum=0)": sm[2],
                "CI95 low": sm[3], "CI95 high": sm[4], "R2": f["r2"]})
    print(f"  {v}: число разрывов по BIC = {m_bic}, по LWZ = {m_lwz}; "
          f"даты (LWZ): {', '.join(f'{d.index[s]:%Y-%m}' for s in br[m_lwz][1]) or 'нет'}")
bpt = pd.DataFrame(rows)
seg = pd.DataFrame(seg_rows)
bpt.round(4).to_csv(C.TAB / "breaks_multiple.csv", index=False)
seg.round(4).to_csv(C.TAB / "breaks_segments.csv", index=False)
print(bpt.round(3).to_string(index=False))
print("\nОценки по режимам, выбранным LWZ:")
print(seg[seg["Selected by"] == "LWZ"].drop(columns=["Selected by"]).round(3).to_string(index=False))

# --------------------------------------------------------------------------
# 4. Оценки по режимам и тест равенства
# --------------------------------------------------------------------------
rows = []
for v, d in data.items():
    for set_name, regs in REGIME_SETS.items():
        parts = []
        for label, a, b in regs:
            sub = d.loc[a:b]
            if len(sub) < 15:
                continue
            f = U.ols(sub["g"].values, sub.drop(columns="g").values)
            V = U.cov_hac(f)
            k = f["k"]
            sm = U.lincom(f["b"], V, slope_sum_row(k), f["n"] - f["k"])
            row = {"Variant": v, "Set": set_name, "Regime": label, "N": f["n"],
                   "R2": f["r2"], "beta_0": f["b"][1],
                   "p (b0)": 2 * stats.t.sf(abs(f["b"][1] / np.sqrt(V[1, 1])), f["n"] - k),
                   "Sum beta": sm[0], "se": sm[1], "p (sum=0)": sm[2],
                   "CI95 low": sm[3], "CI95 high": sm[4]}
            parts.append((sub, row))
        if len(parts) == 2:
            # совместная регрессия с раздельными коэффициентами
            dd = pd.concat([p[0] for p in parts])
            k = dd.shape[1] - 1
            D1 = np.r_[np.ones(len(parts[0][0])), np.zeros(len(parts[1][0]))]
            Xs = dd.drop(columns="g").values
            Z = np.column_stack([Xs * D1[:, None], Xs * (1 - D1)[:, None]])
            f = U.ols(dd["g"].values, Z)
            V = U.cov_hac(f)
            R_sl = np.zeros((k - 1, 2 * k))
            for j in range(1, k):
                R_sl[j - 1, j] = 1.0
                R_sl[j - 1, k + j] = -1.0
            R_sum = np.concatenate([slope_sum_row(k), -slope_sum_row(k)])
            w_sl, dfs, p_sl = U.wald(f["b"], V, R_sl)
            eq = U.lincom(f["b"], V, R_sum, f["n"] - f["k"])
            _, pb_sl = U.wald_boot(dd["g"].values, Z, R_sl, B=N_BOOT_WALD)
            _, pb_sum = U.wald_boot(dd["g"].values, Z, R_sum.reshape(1, -1), B=N_BOOT_WALD)
            for _, row in parts:
                row["Wald equal slopes"] = w_sl
                row["p (equal slopes)"] = p_sl
                row["Diff in sum beta"] = eq[0]
                row["p (equal sums)"] = eq[2]
                row["p boot (equal slopes)"] = pb_sl
                row["p boot (equal sums)"] = pb_sum
        rows.extend(p[1] for p in parts)
reg = pd.DataFrame(rows)
reg.round(4).to_csv(C.TAB / "breaks_regimes.csv", index=False)
print("\n4. Оценки по режимам (HAC) и тест равенства между режимами")
print(reg.round(3).to_string(index=False))

# --------------------------------------------------------------------------
# 5. Скользящая регрессия (baseline)
# --------------------------------------------------------------------------
d = data["baseline"]
y_all, X_all = d["g"].values, d.drop(columns="g").values
k = X_all.shape[1]
roll_rows = []
for end in range(ROLL_WIN, len(d) + 1):
    a = end - ROLL_WIN
    f = U.ols(y_all[a:end], X_all[a:end])
    V = U.cov_hc(f)
    sm = U.lincom(f["b"], V, slope_sum_row(k), f["n"] - f["k"])
    roll_rows.append({"End": d.index[end - 1], "beta_0": f["b"][1],
                      "se b0": np.sqrt(V[1, 1]), "Sum beta": sm[0], "se": sm[1],
                      "CI95 low": sm[3], "CI95 high": sm[4], "R2": f["r2"]})
roll = pd.DataFrame(roll_rows).set_index("End")
roll.round(4).to_csv(C.TAB / "breaks_rolling.csv")
print(f"\n5. Скользящие оценки (окно {ROLL_WIN} мес.): sum beta от "
      f"{roll['Sum beta'].min():+.2f} до {roll['Sum beta'].max():+.2f}")

# --------------------------------------------------------------------------
# Рисунки
# --------------------------------------------------------------------------
print("\nРисунки:")


def add_events(ax):
    for date in C.EVENTS:
        ax.axvline(pd.Timestamp(date), color="grey", ls=":", lw=1)


def fmt_dates(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


# Рис. 9. Траектория F-статистики supF
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
for ax, label in zip(axes, ("slopes only", "all coefficients")):
    idx, res = qa_cache[("baseline", label)]
    dates = idx[res["taus"]]
    ax.plot(dates, res["F"], color="black", lw=1.4, label="baseline")
    if ("ex-COVID", label) in qa_cache:
        idx2, res2 = qa_cache[("ex-COVID", label)]
        ax.plot(idx2[res2["taus"]], res2["F"], color="dimgrey", lw=1.4, ls="--",
                label="ex-COVID")
    ax.axhline(res["crit95"], color="black", ls=":", lw=1,
               label=f"5% bootstrap crit. ({res['crit95']:.1f})")
    add_events(ax)
    fmt_dates(ax)
    ax.set_title(f"supF path: break in {label}")
    ax.set_ylabel("F statistic")
    ax.legend(fontsize=8, loc="upper left")
C.save_fig(fig, "fig9_break_F_path")

# Рис. 10. Скользящие коэффициенты
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axes[0].plot(roll.index, roll["Sum beta"], color="black", lw=1.6)
axes[0].fill_between(roll.index, roll["CI95 low"], roll["CI95 high"], color="grey", alpha=0.3)
axes[0].axhline(0, color="grey", lw=0.8)
axes[0].set_title(f"Rolling {ROLL_WIN}-month estimates (95% HC band)")
axes[0].set_ylabel("Sum of beta_i")
axes[1].plot(roll.index, roll["beta_0"], color="black", lw=1.6)
axes[1].fill_between(roll.index, roll["beta_0"] - 1.96 * roll["se b0"],
                     roll["beta_0"] + 1.96 * roll["se b0"], color="grey", alpha=0.3)
axes[1].axhline(0, color="grey", lw=0.8)
axes[1].set_ylabel("beta_0 (same month)")
for ax in axes:
    add_events(ax)
fmt_dates(axes[1])
C.save_fig(fig, "fig10_rolling_betas")

# Рис. 11. Суммарный эффект по режимам Bai-Perron (LWZ)
fig, ax = plt.subplots(figsize=(9, 4))
for v, style in (("baseline", ("black", "-")), ("ex-COVID", ("dimgrey", "--"))):
    s = seg[(seg["Variant"] == v) & (seg["Selected by"] == "LWZ")]
    d_v = bp_cache[v][0]
    pos = 0
    for _, row in s.iterrows():
        a = d_v.index[pos]
        b = d_v.index[pos + int(row["N"]) - 1]
        ax.hlines(row["Sum beta"], a, b, color=style[0], ls=style[1], lw=2.2,
                  label=v if pos == 0 else None)
        ax.fill_between([a, b], row["CI95 low"], row["CI95 high"], color=style[0], alpha=0.12)
        pos += int(row["N"])
ax.axhline(0, color="grey", lw=0.8)
add_events(ax)
fmt_dates(ax)
ax.set_ylabel("Sum of beta_i by regime")
ax.set_title("Regimes selected by LWZ (Bai-Perron-type), 95% HAC bands")
ax.legend()
C.save_fig(fig, "fig11_segment_betas")

print("\nГотово. Таблицы: output/tables/breaks_*.csv")
