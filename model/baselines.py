"""
Baseline comparison.

The question this answers: is the 35-feature model actually earning its
complexity, or would something far simpler do as well?

Four predictors, same held-out matches, same metrics:

  1. Coin flip           -- the floor. Anything must beat this.
  2. Higher rank wins    -- the trivial predictor. In tennis this is
                            surprisingly strong, around 65% accuracy, and a lot
                            of published models quietly fail to beat it.
  3. Elo                 -- one number per player, updated after each match.
                            The standard baseline in tennis modelling.
  4. Surface Elo         -- same, per surface.
  5. The full model      -- 35 features, gradient boosted, calibrated.
  6. The market          -- closing odds with the vig removed, where available.

If (5) doesn't clearly beat (2) and (3), the extra features are decoration.
That is a real possible outcome and worth finding out before adding more.

Run: python -m model.baselines
"""

from __future__ import annotations

import sqlite3
from datetime import date

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .elo import as_of as elo_as_of
from .elo import win_probability
from .features import FEATURE_NAMES
from .train import build_dataset, load_model, predict_calibrated, time_split

CLIP = (0.02, 0.98)


def _score(name: str, y, p) -> dict:
    p = np.clip(np.asarray(p, dtype=float), *CLIP)
    y = np.asarray(y)
    return {
        "predictor": name,
        "n": int(len(y)),
        "accuracy": float(((p > 0.5).astype(int) == y).mean()),
        "auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
    }


def rank_probability(r1: int | None, r2: int | None) -> float:
    """
    Trivial predictor, softened.

    Hard 0/1 on 'higher rank wins' would be brutally punished by log loss on
    every upset, which flatters the alternatives unfairly. A gentle logistic on
    the log-rank gap is the fair version of the same idea.
    """
    if not r1 or not r2:
        return 0.5
    gap = np.log(r2) - np.log(r1)          # positive => p1 better ranked
    return float(1.0 / (1.0 + np.exp(-0.72 * gap)))


def devig(odds_w: float, odds_l: float) -> tuple[float, float]:
    iw, il = 1 / odds_w, 1 / odds_l
    t = iw + il
    return iw / t, il / t


def compare(conn: sqlite3.Connection, start: str, end: str,
            model_path: str = "model.pkl") -> list[dict]:
    X, y, keys, dates = build_dataset(conn, start, end, verbose=True)
    if len(X) < 300:
        raise ValueError(f"only {len(X)} rows; load more data first")

    tr, cal, te = time_split(dates)
    idx = range(te.start, te.stop)
    y_te = y[te]

    print(f"\n  evaluating on {len(y_te)} held-out matches "
          f"({dates[te.start]} to {dates[-1]})\n")

    # feature columns we can read the simple baselines straight out of
    col = {n: i for i, n in enumerate(FEATURE_NAMES)}
    p_elo = [1 / (1 + np.exp(-X[i][col["d_elo"]] / 173.7)) for i in idx]
    p_elo_s = [1 / (1 + np.exp(-X[i][col["d_elo_surface"]] / 173.7)) for i in idx]
    # d_log_rank is already ln(r2)-ln(r1)
    p_rank = [1 / (1 + np.exp(-0.72 * X[i][col["d_log_rank"]])) for i in idx]

    results = [
        _score("coin flip", y_te, np.full(len(y_te), 0.5)),
        _score("higher rank", y_te, p_rank),
        _score("elo", y_te, p_elo),
        _score("elo (surface)", y_te, p_elo_s),
    ]

    # the trained model
    try:
        m = load_model(model_path)
        p_model = predict_calibrated(m["model"], X[te])
        results.append(_score("full model", y_te, p_model))
    except (FileNotFoundError, ValueError) as exc:
        print(f"  (skipping full model: {exc})")

    # the market, on the subset that has odds
    mp, my = [], []
    for j, i in enumerate(idx):
        row = conn.execute(
            "SELECT winner_odds, loser_odds FROM closing_odds WHERE match_key = ?",
            (keys[i],),
        ).fetchone()
        if not row or not row["winner_odds"] or not row["loser_odds"]:
            continue
        fair_w, _ = devig(row["winner_odds"], row["loser_odds"])
        # label 1 means p1 was the actual winner
        mp.append(fair_w if y_te[j] == 1 else 1 - fair_w)
        my.append(y_te[j])
    if len(mp) > 50:
        results.append(_score(f"market (n={len(mp)})", my, mp))

    return results


def main():
    from datetime import timedelta

    from ingest.db import connect
    conn = connect("tennis.db")
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=1095)).isoformat()

    rows = compare(conn, start, end)

    print(f"  {'predictor':<20} {'n':>6} {'acc':>7} {'auc':>7} "
          f"{'brier':>8} {'logloss':>9}")
    print("  " + "-" * 62)
    for r in rows:
        print(f"  {r['predictor']:<20} {r['n']:>6} {r['accuracy']:>7.3f} "
              f"{r['auc']:>7.3f} {r['brier']:>8.4f} {r['log_loss']:>9.4f}")

    print()
    simple = [r for r in rows if r["predictor"] in
              ("higher rank", "elo", "elo (surface)")]
    full = next((r for r in rows if r["predictor"] == "full model"), None)
    if full and simple:
        best = min(simple, key=lambda r: r["brier"])
        delta = best["brier"] - full["brier"]
        if delta > 0.002:
            print(f"  The full model beats the best simple baseline "
                  f"({best['predictor']}) by {delta:.4f} Brier.")
        elif delta < -0.002:
            print(f"  {best['predictor']} BEATS the full model by "
                  f"{-delta:.4f} Brier. The extra features are not earning "
                  f"their keep.")
        else:
            print(f"  The full model and {best['predictor']} are within noise "
                  f"of each other. The extra complexity is buying little.")


if __name__ == "__main__":
    main()
