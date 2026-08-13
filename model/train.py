"""
Model training.

Two decisions that matter more than the algorithm choice:

1. RANDOMIZED player order. If p1 is always the winner, the label is trivially
   predictable and the model learns nothing. Each match is emitted once with a
   coin-flip on which player is p1, and the label follows.

2. ISOTONIC CALIBRATION on a held-out slice. A gradient booster's raw scores
   are not probabilities. For EV you need probabilities -- a model that ranks
   well but is miscalibrated will size every bet wrong.

Time-ordered splits throughout. Random k-fold on time-series data leaks the
future into training and inflates every number you report.
"""

from __future__ import annotations

import json
import pickle
import random
import sqlite3
from datetime import date

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .features import FEATURE_NAMES, build_features

MIN_MATCHES_PER_PLAYER = 5   # below this, features are mostly defaults

# Isotonic regression is a step function: with limited calibration data it
# happily outputs exactly 0.0 and 1.0. That is never a defensible claim about a
# tennis match, it makes log loss explode on the rare confident miss, and it
# implies infinite Kelly confidence downstream. Clip on the way out.
PROB_FLOOR, PROB_CEIL = 0.02, 0.98

# Isotonic needs data to be smooth. Below this, Platt scaling (sigmoid) is the
# better-behaved choice.
ISOTONIC_MIN_SAMPLES = 1500


def pick_calibration_method(n_calib: int) -> str:
    return "isotonic" if n_calib >= ISOTONIC_MIN_SAMPLES else "sigmoid"


def predict_calibrated(model, X) -> np.ndarray:
    """The ONLY way predictions should leave this codebase."""
    p = model.predict_proba(X)[:, 1]
    return np.clip(p, PROB_FLOOR, PROB_CEIL)


def build_dataset(conn: sqlite3.Connection, start: str, end: str,
                  seed: int = 42, verbose: bool = True) -> tuple:
    """
    Materialize a training matrix over matches in [start, end).

    Slow by design -- features are computed per match with as_of set to that
    match's date, which is the only way to guarantee no leakage. For a few tens
    of thousands of matches this is minutes, not hours. Cache the output.
    """
    rng = random.Random(seed)
    rows = conn.execute(
        """SELECT match_key, match_date, surface, tour_level, winner_id, loser_id
           FROM matches
           WHERE match_date >= ? AND match_date < ?
             AND winner_id IS NOT NULL AND loser_id IS NOT NULL
           ORDER BY match_date""",
        (start, end),
    ).fetchall()

    X, y, keys, dates = [], [], [], []
    skipped = 0

    for i, r in enumerate(rows):
        if verbose and i % 2000 == 0 and i:
            print(f"    {i}/{len(rows)}")

        # p1 is a coin flip -- see docstring
        if rng.random() < 0.5:
            p1, p2, label = r["winner_id"], r["loser_id"], 1
        else:
            p1, p2, label = r["loser_id"], r["winner_id"], 0

        fb = build_features(conn, p1, p2, r["surface"], r["match_date"], r["tour_level"])
        m = fb["meta"]
        if m["p1_gap"] or m["p2_gap"]:
            skipped += 1
            continue
        if fb["features"]["min_season_matches"] < MIN_MATCHES_PER_PLAYER:
            skipped += 1
            continue

        X.append([fb["features"][n] for n in FEATURE_NAMES])
        y.append(label)
        keys.append(r["match_key"])
        dates.append(r["match_date"])

    if verbose:
        print(f"    built {len(X)} rows, skipped {skipped} (insufficient history)")
    return np.array(X, dtype=float), np.array(y), keys, dates


def time_split(dates: list[str], train_frac: float = 0.70, cal_frac: float = 0.15):
    """Chronological train / calibration / test indices."""
    n = len(dates)
    i_tr = int(n * train_frac)
    i_cal = int(n * (train_frac + cal_frac))
    return slice(0, i_tr), slice(i_tr, i_cal), slice(i_cal, n)


def train(conn: sqlite3.Connection, start: str, end: str,
          out_path: str = "model.pkl", seed: int = 42) -> dict:
    print("  building dataset...")
    X, y, keys, dates = build_dataset(conn, start, end, seed=seed)
    if len(X) < 200:
        raise ValueError(f"Only {len(X)} usable rows -- need more history before training.")

    tr, cal, te = time_split(dates)
    print(f"  train={tr.stop - tr.start} calib={cal.stop - cal.start} test={te.stop - te.start}")

    base = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=5,
        min_samples_leaf=40, l2_regularization=1.0, random_state=seed,
    )
    base.fit(X[tr], y[tr])

    # calibrate on a held-out slice the base model never saw
    method = pick_calibration_method(cal.stop - cal.start)
    print(f"  calibration method: {method}")
    model = CalibratedClassifierCV(FrozenEstimator(base), method=method)
    model.fit(X[cal], y[cal])

    p_test = predict_calibrated(model, X[te])
    metrics = {
        "n_train": int(tr.stop - tr.start),
        "n_calib": int(cal.stop - cal.start),
        "calibration_method": method,
        "n_test": int(te.stop - te.start),
        "test_start": dates[te.start] if te.start < len(dates) else None,
        "test_end": dates[-1] if dates else None,
        "auc": float(roc_auc_score(y[te], p_test)),
        "brier": float(brier_score_loss(y[te], p_test)),
        "log_loss": float(log_loss(y[te], p_test)),
        "accuracy": float(((p_test > 0.5).astype(int) == y[te]).mean()),
        # the baselines that make the numbers above mean something
        "brier_coinflip": float(brier_score_loss(y[te], np.full(len(y[te]), 0.5))),
        "brier_base_rate": float(brier_score_loss(
            y[te], np.full(len(y[te]), y[tr].mean()))),
    }

    with open(out_path, "wb") as f:
        pickle.dump({"model": model, "features": FEATURE_NAMES,
                     "prob_clip": [PROB_FLOOR, PROB_CEIL],
                     "metrics": metrics, "trained_at": date.today().isoformat()}, f)

    print(json.dumps(metrics, indent=2))
    return metrics


def calibration_table(y_true, p_pred, bins: int = 10) -> list[dict]:
    """
    Predicted vs observed frequency per bucket.

    This is the plot to put in your README. 'My model is well calibrated' is a
    claim; this table is the evidence.
    """
    out = []
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p_pred >= edges[i]) & (p_pred < edges[i + 1] if i < bins - 1 else p_pred <= 1.0)
        if m.sum() == 0:
            continue
        out.append({
            "bucket": f"{edges[i]:.1f}-{edges[i+1]:.1f}",
            "n": int(m.sum()),
            "predicted": float(p_pred[m].mean()),
            "observed": float(y_true[m].mean()),
        })
    return out


def load_model(path: str = "model.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
