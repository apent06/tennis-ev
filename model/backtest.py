"""
Walk-forward backtest against closing odds.

The honest framing: closing odds minus vig is a hard baseline. Most models that
reproduce public data land at or below it. The point of this file is to find out
which side of that line you're on, and to be able to say so precisely.

What's measured:
  - calibration (are the probabilities real?)
  - Brier / log loss vs the market's own implied probabilities
  - ROI at various edge thresholds, with fractional Kelly sizing
  - the same, split by tour level -- edges usually live at Challenger level
    if they live anywhere

Walk-forward, never random split: train on everything before a cutoff, test on
the window after, roll forward, repeat.
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss

from .features import FEATURE_NAMES
from .train import (build_dataset, calibration_table,
                     pick_calibration_method, predict_calibrated)

KELLY_FRACTION = 0.25    # quarter Kelly. Full Kelly is far too aggressive here.
MAX_STAKE = 0.02         # never more than 2% of bank on one match


def devig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Convert a two-way price pair to fair probabilities by removing the vig."""
    ia, ib = 1 / odds_a, 1 / odds_b
    total = ia + ib
    return ia / total, ib / total


def kelly_stake(p: float, odds: float) -> float:
    """Fractional Kelly, capped. Returns fraction of bank."""
    b = odds - 1
    if b <= 0:
        return 0.0
    edge = p * b - (1 - p)
    if edge <= 0:
        return 0.0
    return min(MAX_STAKE, KELLY_FRACTION * edge / b)


def _odds_for(conn: sqlite3.Connection, match_key: str) -> tuple | None:
    r = conn.execute(
        "SELECT winner_odds, loser_odds FROM closing_odds WHERE match_key=?",
        (match_key,),
    ).fetchone()
    return (r["winner_odds"], r["loser_odds"]) if r else None


def walk_forward(conn: sqlite3.Connection, start: str, end: str,
                 window_days: int = 90, min_train: int = 800,
                 edge_threshold: float = 0.04, seed: int = 42) -> dict:
    """
    Roll a train/test boundary forward through time in `window_days` steps.
    Returns aggregate metrics plus a per-tour-level ROI breakdown.
    """
    print("  building full dataset once (features are as-of, so this is safe)...")
    X, y, keys, dates = build_dataset(conn, start, end, seed=seed, verbose=True)
    if len(X) < min_train + 100:
        raise ValueError(f"Only {len(X)} rows; need at least {min_train + 100}.")

    dates_arr = np.array(dates)
    d0 = date.fromisoformat(dates[min_train])
    d_end = date.fromisoformat(dates[-1])

    all_p, all_y, all_keys = [], [], []
    cursor = d0

    while cursor < d_end:
        nxt = cursor + timedelta(days=window_days)
        tr_mask = dates_arr < cursor.isoformat()
        te_mask = (dates_arr >= cursor.isoformat()) & (dates_arr < nxt.isoformat())

        if tr_mask.sum() < min_train or te_mask.sum() < 20:
            cursor = nxt
            continue

        # hold out the tail of train for calibration
        tr_idx = np.where(tr_mask)[0]
        split = int(len(tr_idx) * 0.85)
        fit_idx, cal_idx = tr_idx[:split], tr_idx[split:]

        base = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.06, max_depth=5,
            min_samples_leaf=40, l2_regularization=1.0, random_state=seed,
        )
        base.fit(X[fit_idx], y[fit_idx])
        clf = CalibratedClassifierCV(
            FrozenEstimator(base), method=pick_calibration_method(len(cal_idx)))
        clf.fit(X[cal_idx], y[cal_idx])

        p = predict_calibrated(clf, X[te_mask])
        all_p.extend(p.tolist())
        all_y.extend(y[te_mask].tolist())
        all_keys.extend([keys[i] for i in np.where(te_mask)[0]])
        print(f"    {cursor} -> {nxt}: train={tr_mask.sum()} test={te_mask.sum()}")
        cursor = nxt

    p_arr, y_arr = np.array(all_p), np.array(all_y)
    if len(p_arr) == 0:
        raise ValueError("No test folds produced. Widen the date range.")

    out = {
        "n_predictions": int(len(p_arr)),
        "accuracy": float(((p_arr > 0.5).astype(int) == y_arr).mean()),
        "brier": float(brier_score_loss(y_arr, p_arr)),
        "log_loss": float(log_loss(y_arr, p_arr)),
        "brier_coinflip": float(brier_score_loss(y_arr, np.full(len(y_arr), 0.5))),
        "calibration": calibration_table(y_arr, p_arr),
    }
    out.update(_market_comparison(conn, all_keys, p_arr, y_arr))
    out.update(_betting_sim(conn, all_keys, p_arr, y_arr, edge_threshold))
    return out


def _market_comparison(conn, keys, p_arr, y_arr) -> dict:
    """Head-to-head against the market's own de-vigged probabilities."""
    mp, mm, my = [], [], []
    for k, p, yy in zip(keys, p_arr, y_arr):
        o = _odds_for(conn, k)
        if not o or not o[0] or not o[1]:
            continue
        fair_w, _ = devig_two_way(o[0], o[1])
        # our p is P(p1 wins); label 1 means p1 was the actual winner
        market_p1 = fair_w if yy == 1 else 1 - fair_w
        mp.append(p)
        mm.append(market_p1)
        my.append(yy)

    if len(mp) < 50:
        return {"market_comparison": "insufficient odds coverage"}

    mp, mm, my = np.array(mp), np.array(mm), np.array(my)
    return {"market_comparison": {
        "n_with_odds": int(len(mp)),
        "model_brier": float(brier_score_loss(my, mp)),
        "market_brier": float(brier_score_loss(my, mm)),
        "model_log_loss": float(log_loss(my, mp)),
        "market_log_loss": float(log_loss(my, mm)),
        "beats_market": bool(brier_score_loss(my, mp) < brier_score_loss(my, mm)),
    }}


def _betting_sim(conn, keys, p_arr, y_arr, edge_threshold: float) -> dict:
    """
    Flat-stake and Kelly simulation over bets that clear the edge threshold.
    Reported per tour level, because that's where the answer usually differs.
    """
    by_level: dict[str, dict] = {}
    bank = 1.0
    n_bets = staked = returned = 0

    for k, p, yy in zip(keys, p_arr, y_arr):
        o = _odds_for(conn, k)
        if not o or not o[0] or not o[1]:
            continue
        row = conn.execute("SELECT tour_level FROM matches WHERE match_key=?", (k,)).fetchone()
        level = row["tour_level"] if row else "?"

        # which side do we think is value? label 1 = p1 won, so p1's price is
        # winner_odds when yy==1 else loser_odds
        p1_odds = o[0] if yy == 1 else o[1]
        p2_odds = o[1] if yy == 1 else o[0]

        for side_p, side_odds, won in ((p, p1_odds, yy == 1),
                                       (1 - p, p2_odds, yy == 0)):
            implied = 1 / side_odds
            edge = side_p - implied
            if edge < edge_threshold:
                continue
            stake = kelly_stake(side_p, side_odds)
            if stake <= 0:
                continue
            n_bets += 1
            staked += stake
            payout = stake * side_odds if won else 0.0
            returned += payout
            bank += payout - stake

            b = by_level.setdefault(level, {"bets": 0, "staked": 0.0, "returned": 0.0})
            b["bets"] += 1
            b["staked"] += stake
            b["returned"] += payout

    for lvl, b in by_level.items():
        b["roi"] = (b["returned"] - b["staked"]) / b["staked"] if b["staked"] else 0.0

    return {"betting": {
        "edge_threshold": edge_threshold,
        "n_bets": n_bets,
        "roi": (returned - staked) / staked if staked else 0.0,
        "final_bank": round(bank, 4),
        "by_tour_level": by_level,
        "note": "Backtested edges shrink against live markets. Treat as a "
                "calibration check, not a forecast of returns.",
    }}
