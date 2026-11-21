"""
Небольшой набор эконометрических функций на numpy/scipy (без statsmodels).

Нужен для скриптов 05 (структурные сдвиги) и 06 (асимметрия): там требуются
робастные тесты, supF с бутстрэпом и поиск нескольких разрывов (Bai-Perron),
которых нет в statsmodels в готовом виде.
"""
import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------
# OLS и ковариационные матрицы
# --------------------------------------------------------------------------
def ols(y, X):
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float)
    n, k = X.shape
    A = np.linalg.pinv(X.T @ X)
    b = A @ (X.T @ y)
    e = y - X @ b
    tss = float(((y - y.mean()) ** 2).sum())
    rss = float(e @ e)
    return {"b": b, "e": e, "rss": rss, "n": n, "k": k, "A": A, "X": X, "y": y,
            "r2": 1 - rss / tss if tss > 0 else np.nan}


def nw_lags(n):
    """Правило Ньюи-Веста: floor(4 (n/100)^(2/9))."""
    return max(int(np.floor(4 * (n / 100) ** (2 / 9))), 1)


def cov_hc(f):
    """Робастная к гетероскедастичности (HC1)."""
    X, e, A, n, k = f["X"], f["e"], f["A"], f["n"], f["k"]
    S = X * e[:, None]
    return (n / max(n - k, 1)) * A @ (S.T @ S) @ A


def cov_hac(f, L=None):
    """Ньюи-Вест (ядро Бартлетта), строки должны идти в порядке времени."""
    X, e, A, n, k = f["X"], f["e"], f["A"], f["n"], f["k"]
    L = L or nw_lags(n)
    S = X * e[:, None]
    meat = S.T @ S
    for l in range(1, min(L, n - 1) + 1):
        w = 1 - l / (L + 1)
        G = S[l:].T @ S[:-l]
        meat = meat + w * (G + G.T)
    return (n / max(n - k, 1)) * A @ meat @ A


# --------------------------------------------------------------------------
# Тесты на линейные ограничения
# --------------------------------------------------------------------------
def wald(b, V, R, r0=None):
    """Wald-тест H0: R b = r0. Возвращает (статистика, df, p-value по chi2)."""
    R = np.atleast_2d(np.asarray(R, float))
    r0 = np.zeros(R.shape[0]) if r0 is None else np.asarray(r0, float)
    d = R @ b - r0
    M = R @ V @ R.T
    stat = float(d @ np.linalg.pinv(M) @ d)
    df = R.shape[0]
    return stat, df, float(stats.chi2.sf(stat, df))


def lincom(b, V, R, df_resid):
    """Линейная комбинация R b: оценка, ст. ошибка, p-value, 95% ДИ (t-распр.)."""
    R = np.asarray(R, float).ravel()
    est = float(R @ b)
    se = float(np.sqrt(max(R @ V @ R, 0.0)))
    if se == 0 or not np.isfinite(se):
        return est, se, np.nan, np.nan, np.nan
    t = est / se
    p = float(2 * stats.t.sf(abs(t), max(df_resid, 1)))
    crit = stats.t.ppf(0.975, max(df_resid, 1))
    return est, se, p, est - crit * se, est + crit * se


def coef_table(f, V, names):
    se = np.sqrt(np.diag(V))
    t = f["b"] / se
    p = 2 * stats.t.sf(np.abs(t), max(f["n"] - f["k"], 1))
    return pd.DataFrame({"Term": names, "Coef": f["b"], "se": se, "t": t, "p-value": p})


# --------------------------------------------------------------------------
# supF (Quandt-Andrews) с «диким» бутстрэпом при фиксированных регрессорах
# --------------------------------------------------------------------------
def supf_paths(Y, X, slope_idx, trim=0.15):
    """F-статистика разрыва в коэффициентах slope_idx для каждой даты tau.
    Y: n x m (m=1 для данных, m=B для бутстрэпа). tau — первое наблюдение
    второго режима, tau in [trim*n, (1-trim)*n)."""
    n, k = X.shape
    lo = int(np.ceil(trim * n))
    taus = np.arange(lo, n - lo)
    q = len(slope_idx)
    yy = (Y ** 2).sum(0)
    Br = np.linalg.pinv(X) @ Y
    rss_r = yy - (Br * (X.T @ Y)).sum(0)
    F = np.empty((len(taus), Y.shape[1]))
    for i, tau in enumerate(taus):
        D = np.zeros(n)
        D[tau:] = 1.0
        Z = np.column_stack([X, X[:, slope_idx] * D[:, None]])
        B = np.linalg.pinv(Z) @ Y
        rss_u = yy - (B * (Z.T @ Y)).sum(0)
        F[i] = ((rss_r - rss_u) / q) / (rss_u / (n - Z.shape[1]))
    return taus, F


def supf_boot(y, X, slope_idx, trim=0.15, B=999, seed=12345):
    y = np.asarray(y, float).ravel()
    taus, F0 = supf_paths(y[:, None], X, slope_idx, trim)
    F0 = F0[:, 0]
    sup0, tau0 = float(F0.max()), int(taus[F0.argmax()])
    fr = ols(y, X)
    rng = np.random.default_rng(seed)
    Ystar = (X @ fr["b"])[:, None] + fr["e"][:, None] * rng.standard_normal((len(y), B))
    _, Fb = supf_paths(Ystar, X, slope_idx, trim)
    supb = Fb.max(0)
    return {"taus": taus, "F": F0, "sup": sup0, "tau": tau0,
            "p": float((1 + (supb >= sup0).sum()) / (B + 1)),
            "crit90": float(np.quantile(supb, 0.90)),
            "crit95": float(np.quantile(supb, 0.95)),
            "crit99": float(np.quantile(supb, 0.99))}


# --------------------------------------------------------------------------
# Несколько разрывов (Bai-Perron, чистая структурная смена, динамическое
# программирование по минимуму суммы квадратов остатков)
# --------------------------------------------------------------------------
def bp_breaks(y, X, h, M):
    """Возвращает {m: (RSS, [индексы начала новых режимов])}, m = 0..M."""
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float)
    n, k = X.shape
    seg = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + h - 1, n):
            Xi, yi = X[i:j + 1], y[i:j + 1]
            b = np.linalg.lstsq(Xi, yi, rcond=None)[0]
            r = yi - Xi @ b
            seg[i, j] = r @ r
    best = np.full((M + 2, n), np.inf)
    arg = np.zeros((M + 2, n), dtype=int)
    best[1] = seg[0]
    for s in range(2, M + 2):
        for j in range(s * h - 1, n):
            for i in range((s - 1) * h, j - h + 2):
                v = best[s - 1][i - 1] + seg[i, j]
                if v < best[s][j]:
                    best[s][j] = v
                    arg[s][j] = i
    out = {0: (float(best[1][n - 1]), [])}
    for m in range(1, M + 1):
        s = m + 1
        if not np.isfinite(best[s][n - 1]):
            continue
        starts, j = [], n - 1
        for ss in range(s, 1, -1):
            i = arg[ss][j]
            starts.append(int(i))
            j = i - 1
        out[m] = (float(best[s][n - 1]), sorted(starts))
    return out


def bp_criteria(rss, n, k, m):
    """BIC и LWZ (Liu-Wu-Zidek) по Bai-Perron (2003); p — число параметров."""
    p = (m + 1) * k + m
    bic = np.log(rss / n) + p * np.log(n) / n
    lwz = np.log(rss / (n - p)) + p * 0.299 * (np.log(n)) ** 2.1 / n
    return bic, lwz


# --------------------------------------------------------------------------
# Робастные оценки (устойчивые к выбросам) и бутстрэп по блокам
# --------------------------------------------------------------------------
def huber(y, X, c=1.345, iters=60):
    """M-оценка Хубера (IRLS); масштаб — MAD остатков."""
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iters):
        e = y - X @ b
        s = max(np.median(np.abs(e - np.median(e))) / 0.6745, 1e-8)
        u = np.abs(e / s)
        w = np.where(u <= c, 1.0, c / np.maximum(u, 1e-12))
        W = np.sqrt(w)
        bn = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)[0]
        done = np.max(np.abs(bn - b)) < 1e-8
        b = bn
        if done:
            break
    return b


def lad(y, X, iters=200, eps=1e-6):
    """Медианная регрессия (LAD) через IRLS."""
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iters):
        e = y - X @ b
        W = np.sqrt(1.0 / np.maximum(np.abs(e), eps))
        bn = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)[0]
        done = np.max(np.abs(bn - b)) < 1e-9
        b = bn
        if done:
            break
    return b


def ols_b(y, X):
    return np.linalg.lstsq(np.asarray(X, float), np.asarray(y, float).ravel(), rcond=None)[0]


def block_boot(y, X, estimator, L=3, B=499, seed=2024):
    """Скользящий блочный бутстрэп пар (y, X). Возвращает B x k оценок."""
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float)
    n = len(y)
    L = max(1, min(L, n))
    nb = int(np.ceil(n / L))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        starts = rng.integers(0, n - L + 1, nb)
        idx = np.concatenate([np.arange(s, s + L) for s in starts])[:n]
        out.append(estimator(y[idx], X[idx]))
    return np.array(out)


# --------------------------------------------------------------------------
# Сравнение прогнозов: Diebold-Mariano и Clark-West
# --------------------------------------------------------------------------
def _lrv(x, L):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    s = x @ x / n
    for l in range(1, min(L, n - 1) + 1):
        s += 2 * (1 - l / (L + 1)) * (x[l:] @ x[:-l]) / n
    return s


def dm_test(e_bench, e_model, L=None):
    """DM: H1 — модель точнее бенчмарка (MSE). Возвращает (stat, p одностор.)."""
    d = np.asarray(e_bench) ** 2 - np.asarray(e_model) ** 2
    n = len(d)
    L = L if L is not None else max(int(round(n ** (1 / 3))), 1)
    v = _lrv(d, L) / n
    if v <= 0:
        return np.nan, np.nan
    stat = d.mean() / np.sqrt(v)
    # поправка Харви-Леборн-Ньюболда: t-распределение с n-1 степенью свободы
    return float(stat), float(stats.t.sf(stat, n - 1))


def cw_test(y, f_bench, f_model, L=None):
    """Clark-West для вложенных моделей: H1 — модель с предикторами точнее."""
    e1 = np.asarray(y) - np.asarray(f_bench)
    e2 = np.asarray(y) - np.asarray(f_model)
    adj = e1 ** 2 - (e2 ** 2 - (np.asarray(f_bench) - np.asarray(f_model)) ** 2)
    n = len(adj)
    L = L if L is not None else max(int(round(n ** (1 / 3))), 1)
    v = _lrv(adj, L) / n
    if v <= 0:
        return np.nan, np.nan
    stat = adj.mean() / np.sqrt(v)
    return float(stat), float(stats.norm.sf(stat))


# --------------------------------------------------------------------------
# Wald-тест с «диким» бутстрэпом (Радемахер) при ограничениях R b = 0
# --------------------------------------------------------------------------
def wald_boot(y, Z, R, B=499, seed=7, cov=None):
    """Wald-статистика с HAC и бутстрэп-p-value (нулевая гипотеза налагается).
    Нужен для малых режимов с тяжёлыми хвостами, где асимптотический chi2 завышает
    число отвержений (проверено Монте-Карло: ~15% при номинальных 5%)."""
    cov = cov or cov_hac
    y = np.asarray(y, float).ravel()
    R = np.atleast_2d(np.asarray(R, float))
    f = ols(y, Z)
    stat = wald(f["b"], cov(f), R)[0]
    A = f["A"]
    adj = A @ R.T @ np.linalg.pinv(R @ A @ R.T) @ (R @ f["b"])
    fit_r = np.asarray(Z, float) @ (f["b"] - adj)
    e_r = y - fit_r
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(B):
        ys = fit_r + e_r * rng.choice([-1.0, 1.0], size=len(y))
        fs = ols(ys, Z)
        cnt += wald(fs["b"], cov(fs), R)[0] >= stat
    return float(stat), float((1 + cnt) / (B + 1))
