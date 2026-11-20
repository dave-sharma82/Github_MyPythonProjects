"""Australian Powerball exploratory analytics; historical draws do not predict fair draws."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from math import comb
import numpy as np
import pandas as pd
from scipy.stats import chisquare, binomtest
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss

COLS = [f"Winning Number {i}" for i in range(1, 8)]
N = 35
K = 7


def load_draws(file):
    raw = pd.read_csv(file)
    required = ["Draw date", "Draw number", *COLS]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")
    raw["date"] = pd.to_datetime(raw["Draw date"], dayfirst=True, errors="coerce")
    selection = raw.loc[raw[COLS].notna().all(axis=1)].copy()
    valid = selection[COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    within = np.isfinite(valid).all(axis=1) & (valid >= 1).all(axis=1) & (valid <= 35).all(axis=1)
    unique = np.array([len(set(row)) == 7 for row in valid], dtype=bool)
    selection = selection.loc[within & unique & selection["date"].notna()].sort_values("date").reset_index(drop=True)
    if selection.empty:
        raise ValueError("No valid draws with seven unique main numbers in the range 1–35.")
    matrix = selection[COLS].to_numpy(dtype=int)
    occurrence = np.zeros((len(selection), 35), dtype=int)
    occurrence[np.arange(len(selection))[:, None], matrix - 1] = 1
    return raw, selection, occurrence


def frequency_table(occ):
    n = len(occ)
    counts = occ.sum(axis=0)
    expected = n * K / N
    # standardized under the binomial marginal null (per number, each draw P=7/35)
    z = (counts - expected) / np.sqrt(n * (K / N) * (1 - K / N))
    return pd.DataFrame({"Number": np.arange(1, 36), "Count": counts, "Frequency %": counts / n * 100,
                         "Expected": expected, "Z score": z})


def pair_table(occ):
    co = occ.T @ occ
    out = [(i + 1, j + 1, int(co[i, j])) for i in range(35) for j in range(i + 1, 35)]
    return pd.DataFrame(out, columns=["First", "Second", "Together"]), co


def draw_features(draws):
    numbers = np.sort(draws[COLS].to_numpy(dtype=int), axis=1)
    data = pd.DataFrame({"Date": draws["date"], "Draw": draws["Draw number"],
                         "Sum": numbers.sum(axis=1), "Odd count": (numbers % 2 == 1).sum(axis=1),
                         "Low (1–17)": (numbers <= 17).sum(axis=1),
                         "Consecutive pairs": (np.diff(numbers, axis=1) == 1).sum(axis=1),
                         "Range": numbers[:, -1] - numbers[:, 0]})
    return data


def since_last(occ):
    out = []
    for j in range(35):
        seen = np.flatnonzero(occ[:, j])
        out.append(len(occ) - 1 - seen[-1] if len(seen) else len(occ))
    return np.array(out)


def rolling_counts(occ, window):
    return occ[-min(window, len(occ)):].sum(axis=0)


def stats_summary(occ):
    counts = occ.sum(axis=0)
    # counts across numbers are dependent within a draw: this chi-square is DESCRIPTIVE,
    # not a calibrated multinomial independence test. Use Monte Carlo null below.
    statistic = float(((counts - counts.mean()) ** 2 / counts.mean()).sum())
    rng = np.random.default_rng(20261120)
    trials = 2500
    null_stats = np.empty(trials)
    for b in range(trials):
        sim = np.zeros(35, dtype=int)
        for row in rng.random((len(occ), 35)).argpartition(7, axis=1)[:, :7]:
            sim[row] += 1
        null_stats[b] = ((sim - counts.mean()) ** 2 / counts.mean()).sum()
    pvalue = (1 + np.count_nonzero(null_stats >= statistic)) / (trials + 1)
    return statistic, pvalue, null_stats


def prepare_model(occ, min_history=52):
    """Predict next draw indicators using only past observations. No leakage."""
    X, Y = [], []
    for t in range(min_history, len(occ)):
        past = occ[:t]
        feat = np.concatenate([past[-8:].mean(0), past[-26:].mean(0), past[-52:].mean(0),
                               np.array([t / len(occ)])])
        X.append(feat)
        Y.append(occ[t])
    return np.asarray(X), np.asarray(Y)


def backtest(occ, alpha=100.0):
    X, Y = prepare_model(occ)
    if len(X) < 70:
        return None
    split = max(40, int(len(X) * 0.7))
    clf = Ridge(alpha=alpha)
    clf.fit(X[:split], Y[:split])
    pred = np.clip(clf.predict(X[split:]), 0.005, 0.995)
    baseline = np.full_like(pred, 0.2)
    model_brier = float(np.mean((pred - Y[split:]) ** 2))
    base_brier = float(np.mean((baseline - Y[split:]) ** 2))
    clf.fit(X, Y)
    latest_feat = np.concatenate([occ[-8:].mean(0), occ[-26:].mean(0), occ[-52:].mean(0), np.array([1.0])])
    model_estimate = np.clip(clf.predict(latest_feat.reshape(1, -1)).ravel(), .02, .5)
    return {"model_brier": model_brier, "baseline_brier": base_brier,
            "test_draws": len(Y) - split, "model_estimate": model_estimate,
            "test_dates_idx": np.arange(52 + split, len(occ)),
            "test_prediction": pred, "test_actual": Y[split:]}


def historical_weights(occ, style="Blended", model=None):
    n = len(occ)
    base = np.ones(35) / 35
    if style == "Uniform / unbiased":
        return base
    # Bayesian shrinkage yields much less aggressive rankings than raw frequencies
    overall = (occ.sum(axis=0) + 50 * .2) / (n + 50)
    recent = (rolling_counts(occ, 52) + 50 * .2) / (min(n, 52) + 50)
    vals = .75 * overall + .25 * recent
    if style == "Experimental ML" and model is not None:
        vals = .65 * vals + .35 * model
    vals = np.maximum(vals, 1e-6)
    vals = vals / vals.sum()
    # temper: small ranking preferences only; never treated as predictive probabilities
    return .65 * base + .35 * vals


def generate_tickets(occ, n_tickets=10, seed=2026, style="Blended", model=None, max_overlap=3,
                     avoid_previous=True):
    rng = np.random.default_rng(seed)
    weights = historical_weights(occ, style, model)
    historical = {tuple(np.flatnonzero(row) + 1) for row in occ} if avoid_previous else set()
    sums = np.flatnonzero(occ).reshape(-1) if False else np.array([int((np.flatnonzero(r) + 1).sum()) for r in occ])
    low_sum, high_sum = np.percentile(sums, [8, 92])
    tickets = []
    for i in range(n_tickets):
        best, best_score = None, -np.inf
        for attempt in range(4500):
            ticket = tuple(sorted(rng.choice(np.arange(1, 36), size=7, replace=False, p=weights)))
            if ticket in historical:
                continue
            a = np.asarray(ticket)
            overlap = max((len(set(ticket) & set(t)) for t in tickets), default=0)
            if overlap > max_overlap:
                continue
            odd = (a % 2).sum()
            consecutive = (np.diff(a) == 1).sum()
            low = (a <= 17).sum()
            value = a.sum()
            # Soft descriptive diversification, not a prediction of next draw.
            score = 0.0
            score -= 2.4 * max(0, abs(odd - 3.5) - 1.5)
            score -= 2.2 * max(0, abs(low - 3.5) - 1.5)
            score -= 2.0 * max(0, consecutive - 2)
            score -= 0.08 * max(0, low_sum - value, value - high_sum)
            score -= 1.3 * overlap
            score += rng.normal(0, .3)
            if score > best_score:
                best, best_score = ticket, score
            if score >= -0.25 and attempt > 100:
                break
        if best is None:
            raise RuntimeError("Could not create enough distinct tickets. Relax max overlap.")
        tickets.append(best)
    return pd.DataFrame([{"Game": i + 1, **{f"N{j+1}": t[j] for j in range(7)},
                          "Sum": sum(t), "Odd": sum(a % 2 for a in t)} for i, t in enumerate(tickets)])


def probability_summary(games=10):
    combos = comb(35, 7)
    return {"seven_combinations": combos, "one_game_seven_match": 1 / combos,
            "ten_games_seven_match": games / combos,
            "one_game_with_single_powerball": 1 / (combos * 20)}
