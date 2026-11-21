"""
Этап 8, часть Б. Асимметрия реакции рубля на рост и падение нефти.

Так как коинтеграции нет (скрипт 03), NARDL в уровнях не применяется.
Используется асимметричная модель с распределёнными лагами в приростах:

    g_t = a + sum_i bp_i * r+_{t-i} + sum_i bn_i * r-_{t-i} + e_t
    r+ = max(r, 0)  (рост нефти),   r- = min(r, 0)  (падение нефти)

Интерпретация:
  * bp: реакция USD/RUB на рост Brent на 1 п.п. (ожидаем отрицательный знак);
  * bn: реакция на падение Brent. Так как r- <= 0, эффект падения на 1 п.п.
    равен -bn, поэтому при bn < 0 падение нефти ОСЛАБЛЯЕТ рубль.
  * Симметрия: sum(bp) = sum(bn) (накопленный эффект) и bp_i = bn_i для
    каждого лага (совместно). Тесты Вальда с HAC-ковариацией.

Все оценки в вариантах baseline и ex-COVID (месяцы март–июнь 2020 и месяцы,
чей лаг попадает в этот шок, исключены): падение Brent на 55% в апреле 2020
лежит целиком в r-, и асимметрия легко возникает из одного этого эпизода.

Таблицы: asym_results.csv, asym_coefficients.csv, asym_dynamic.csv
Рисунки: fig12_asym_dynamic, fig13_asym_by_sample

Запуск:  python 06_asymmetry.py   (после 01_prepare_data.py)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as C
import econ_utils as U

C.apply_plot_style()

P_LIST = [1, 3]     # числа лагов (спецификации)
MIN_DOF = 20        # минимум степеней свободы для оценки

df = pd.read_csv(C.PROC / "monthly.csv", index_col="date", parse_dates=True)
print(f"Загружено {len(df)} наблюдений: {df.index.min():%Y-%m} — {df.index.max():%Y-%m}")

SAMPLES = {"full": (C.START, C.END),
           "pre-2022": (C.START, "2022-02-01"),
           "post-2022": (C.SANCTIONS_2022, C.END)}
VARIANTS = {"baseline": False, "ex-COVID": True}


def make_data(P, drop_covid):
    X = pd.DataFrame({"const": 1.0}, index=df.index)
    for i in range(P + 1):
        X[f"rp_l{i}"] = df["r_pos"].shift(i)
    for i in range(P + 1):
        X[f"rn_l{i}"] = df["r_neg"].shift(i)
    d = pd.concat([df["g"], X], axis=1).dropna()
    if drop_covid:
        bad = pd.Series(False, index=df.index)
        for i in range(P + 1):
            bad |= df["D_covid"].shift(i).fillna(0).astype(bool)
        d = d.loc[~bad.reindex(d.index).values]
    return d


def vec(k, P, kind, upto=None):
    """Вектор, суммирующий rp (kind='p') или rn (kind='n') до лага upto."""
    R = np.zeros(k)
    top = P if upto is None else upto
    base = 1 if kind == "p" else 1 + (P + 1)
    R[base: base + top + 1] = 1.0
    return R


res_rows, coef_rows, dyn_rows = [], [], []
fits = {}
for v, drop in VARIANTS.items():
    for P in P_LIST:
        d_all = make_data(P, drop)
        for sname, (a, b) in SAMPLES.items():
            d = d_all.loc[a:b]
            k = d.shape[1] - 1
            if len(d) - k < MIN_DOF:
                res_rows.append({"Variant": v, "p": P, "Sample": sname, "N": len(d),
                                 "Note": "мало наблюдений — пропущено"})
                continue
            f = U.ols(d["g"].values, d.drop(columns="g").values)
            V = U.cov_hac(f)
            names = list(d.drop(columns="g").columns)
            fits[(v, P, sname)] = (f, V, d)
            ct = U.coef_table(f, V, names)
            ct.insert(0, "Sample", sname)
            ct.insert(0, "p", P)
            ct.insert(0, "Variant", v)
            coef_rows.append(ct)

            dfr = f["n"] - f["k"]
            sp = U.lincom(f["b"], V, vec(k, P, "p"), dfr)
            sn = U.lincom(f["b"], V, vec(k, P, "n"), dfr)
            diff = U.lincom(f["b"], V, vec(k, P, "p") - vec(k, P, "n"), dfr)
            # попарное равенство bp_i = bn_i для всех i
            R_pair = np.zeros((P + 1, k))
            for i in range(P + 1):
                R_pair[i, 1 + i] = 1.0
                R_pair[i, 1 + (P + 1) + i] = -1.0
            w_pair, df_pair, p_pair = U.wald(f["b"], V, R_pair)
            # значимость блоков
            R_p = np.zeros((P + 1, k)); R_n = np.zeros((P + 1, k))
            for i in range(P + 1):
                R_p[i, 1 + i] = 1.0
                R_n[i, 1 + (P + 1) + i] = 1.0
            _, _, p_blk_p = U.wald(f["b"], V, R_p)
            _, _, p_blk_n = U.wald(f["b"], V, R_n)
            res_rows.append({
                "Variant": v, "p": P, "Sample": sname, "N": f["n"], "R2": f["r2"],
                "Sum bp (rises)": sp[0], "se bp": sp[1], "p (bp=0)": sp[2],
                "Sum bn (falls)": sn[0], "se bn": sn[1], "p (bn=0)": sn[2],
                "Effect of +10% Brent, pp": 10 * sp[0],
                "Effect of -10% Brent, pp": -10 * sn[0],
                "Sum bp - Sum bn": diff[0], "p (sums equal)": diff[2],
                "Wald pairwise equal": w_pair, "p (pairwise equal)": p_pair,
                "p (all bp=0)": p_blk_p, "p (all bn=0)": p_blk_n,
                "CI95 bp low": sp[3], "CI95 bp high": sp[4],
                "CI95 bn low": sn[3], "CI95 bn high": sn[4], "Note": ""})
            if sname == "full":
                for h in range(P + 1):
                    cp = U.lincom(f["b"], V, vec(k, P, "p", h), dfr)
                    cn = U.lincom(f["b"], V, vec(k, P, "n", h), dfr)
                    dyn_rows.append({"Variant": v, "p": P, "Horizon": h,
                                     "Cum bp": cp[0], "bp low": cp[3], "bp high": cp[4],
                                     "Cum bn": cn[0], "bn low": cn[3], "bn high": cn[4]})

res = pd.DataFrame(res_rows)
coef = pd.concat(coef_rows, ignore_index=True)
dyn = pd.DataFrame(dyn_rows)
res.round(4).to_csv(C.TAB / "asym_results.csv", index=False)
coef.round(4).to_csv(C.TAB / "asym_coefficients.csv", index=False)
dyn.round(4).to_csv(C.TAB / "asym_dynamic.csv", index=False)

show = ["Variant", "p", "Sample", "N", "Sum bp (rises)", "p (bp=0)", "Sum bn (falls)",
        "p (bn=0)", "Sum bp - Sum bn", "p (sums equal)", "p (pairwise equal)"]
print("\nАсимметричная модель: накопленные эффекты и тесты симметрии (HAC)")
print(res[show].round(3).to_string(index=False))
print("\nЭффект в п.п. изменения USD/RUB при изменении Brent на 10%:")
print(res[["Variant", "p", "Sample", "Effect of +10% Brent, pp", "Effect of -10% Brent, pp"]]
      .round(2).to_string(index=False))

# --------------------------------------------------------------------------
# Рисунки
# --------------------------------------------------------------------------
print("\nРисунки:")
P_DYN = max(P_LIST)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, v in zip(axes, VARIANTS):
    s = dyn[(dyn["Variant"] == v) & (dyn["p"] == P_DYN)]
    if s.empty:
        continue
    ax.plot(s["Horizon"], s["Cum bp"], color="black", marker="o", ms=4,
            label="Oil rises (b+)")
    ax.fill_between(s["Horizon"], s["bp low"], s["bp high"], color="black", alpha=0.12)
    ax.plot(s["Horizon"], s["Cum bn"], color="dimgrey", marker="s", ms=4, ls="--",
            label="Oil falls (b-)")
    ax.fill_between(s["Horizon"], s["bn low"], s["bn high"], color="grey", alpha=0.18)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(s["Horizon"])
    ax.set_xlabel("Horizon, months")
    ax.set_title(f"{v}, full sample")
axes[0].set_ylabel("Cumulative coefficient on r+ / r-")
axes[0].legend(fontsize=8)
fig.suptitle("Cumulative responses of USD/RUB: oil rises vs oil falls (95% HAC bands)")
C.save_fig(fig, "fig12_asym_dynamic")

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
P_BAR = P_LIST[0]
for ax, v in zip(axes, VARIANTS):
    s = res[(res["Variant"] == v) & (res["p"] == P_BAR) & res["N"].notna()
            & res["Sum bp (rises)"].notna()]
    x = np.arange(len(s))
    w = 0.35
    ax.bar(x - w / 2, s["Sum bp (rises)"], w, color="black", label="Oil rises (sum b+)",
           yerr=[s["Sum bp (rises)"] - s["CI95 bp low"], s["CI95 bp high"] - s["Sum bp (rises)"]],
           capsize=3)
    ax.bar(x + w / 2, s["Sum bn (falls)"], w, color="silver", edgecolor="black",
           label="Oil falls (sum b-)",
           yerr=[s["Sum bn (falls)"] - s["CI95 bn low"], s["CI95 bn high"] - s["Sum bn (falls)"]],
           capsize=3)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(s["Sample"])
    ax.set_title(f"{v} (p={P_BAR})")
axes[0].set_ylabel("Sum of coefficients (pp per 1 pp)")
axes[0].legend(fontsize=8)
fig.suptitle("Asymmetry by sample (95% HAC intervals)")
C.save_fig(fig, "fig13_asym_by_sample")

print("\nГотово. Таблицы: output/tables/asym_*.csv")
print("Напоминание: bn < 0 означает, что ПАДЕНИЕ нефти ослабляет рубль (USD/RUB растёт).")
