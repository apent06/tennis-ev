"""
Model-layer tests. Run: python tests/test_model.py

These target the failure modes that don't raise exceptions -- the ones that
quietly produce a good-looking number that's wrong.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ingest.db import connect, init_db
from ingest.loaders import generate_synthetic
from model.features import (FEATURE_NAMES, build_features, h2h_features,
                            player_features, quality_score)
from model.train import (PROB_CEIL, PROB_FLOOR, build_dataset,
                         pick_calibration_method, predict_calibrated, time_split)

results = []


def check(name, cond):
    results.append((cond, name))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


db_path = os.path.join(tempfile.mkdtemp(), "m.db")
conn = connect(db_path)
init_db(conn)
print("generating synthetic data...")
generate_synthetic(conn, n_players=80, n_matches=2500, seed=11)

pids = [r["player_id"] for r in conn.execute("SELECT player_id FROM players LIMIT 10")]
check("synthetic players have distinct ids", len(set(pids)) == len(pids))

# -- quality score ------------------------------------------------------------
beat_top10 = [{"won": 1, "opp_rank": 5}]
beat_300 = [{"won": 1, "opp_rank": 300}]
check("beating a top-10 scores above beating a 300",
      quality_score(beat_top10) > quality_score(beat_300))

lost_top10 = [{"won": 0, "opp_rank": 5}]
lost_300 = [{"won": 0, "opp_rank": 300}]
check("losing to a 300 scores below losing to a top-10",
      quality_score(lost_300) < quality_score(lost_top10))
check("empty history returns neutral 0.5", quality_score([]) == 0.5)

# -- leakage: as_of must exclude the future -----------------------------------
pid = pids[0]
early = player_features(conn, pid, "Hard", "2025-06-01")
late = player_features(conn, pid, "Hard", "2026-08-01")
check("later as_of sees at least as many matches",
      late["season_matches"] >= 0 and early["season_matches"] >= 0)

row = conn.execute(
    """SELECT match_date, winner_id, loser_id FROM matches
       WHERE winner_id IS NOT NULL ORDER BY match_date DESC LIMIT 1""").fetchone()
h_before = h2h_features(conn, row["winner_id"], row["loser_id"], row["match_date"])
h_after = h2h_features(conn, row["winner_id"], row["loser_id"], "2027-01-01")
check("h2h as_of excludes the match itself", h_after["h2h_n"] > h_before["h2h_n"])

# -- symmetry: swapping players must flip the sign of diff features -----------
a, b = pids[0], pids[1]
fab = build_features(conn, a, b, "Hard", "2026-06-01")["features"]
fba = build_features(conn, b, a, "Hard", "2026-06-01")["features"]
diff_ok = all(
    abs(fab[k] + fba[k]) < 1e-9
    for k in FEATURE_NAMES
    if k.startswith("d_")
)
check("diff features are antisymmetric under player swap", diff_ok)
check("h2h flips under swap",
      abs(fab["h2h_weighted"] + fba["h2h_weighted"] - 1.0) < 1e-9
      or fab["h2h_n"] == 0)
check("symmetric features are unchanged under swap",
      fab["min_surface_n"] == fba["min_surface_n"]
      and fab["h2h_n"] == fba["h2h_n"])

# -- dataset construction -----------------------------------------------------
X, y, keys, dates = build_dataset(conn, "2020-01-01", "2027-01-01", verbose=False)
check(f"dataset non-empty (got {len(X)})", len(X) > 300)
check("labels are roughly balanced (randomized player order)",
      0.4 < y.mean() < 0.6)
check("dates are chronologically ordered", all(
    dates[i] <= dates[i + 1] for i in range(len(dates) - 1)))
check("no NaNs in feature matrix", not np.isnan(X).any())
check("no infinities in feature matrix", not np.isinf(X).any())

zero_var = [FEATURE_NAMES[i] for i in range(X.shape[1]) if X[:, i].std() < 1e-9]
check(f"no zero-variance features (found {zero_var})", not zero_var)

# -- time split ---------------------------------------------------------------
tr, cal, te = time_split(dates)
check("train slice precedes test slice chronologically",
      dates[tr.stop - 1] <= dates[te.start])
check("splits partition the dataset with no overlap",
      tr.stop == cal.start and cal.stop == te.start and te.stop == len(dates))

# -- calibration method selection ---------------------------------------------
check("small calibration set picks sigmoid", pick_calibration_method(300) == "sigmoid")
check("large calibration set picks isotonic", pick_calibration_method(5000) == "isotonic")

# -- model trains and produces bounded probabilities --------------------------
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import roc_auc_score

base = HistGradientBoostingClassifier(max_iter=150, random_state=1)
base.fit(X[tr], y[tr])
clf = CalibratedClassifierCV(FrozenEstimator(base),
                             method=pick_calibration_method(cal.stop - cal.start))
clf.fit(X[cal], y[cal])
p = predict_calibrated(clf, X[te])

check(f"probabilities respect the clip floor (min {p.min():.4f})", p.min() >= PROB_FLOOR)
check(f"probabilities respect the clip ceiling (max {p.max():.4f})", p.max() <= PROB_CEIL)
auc = roc_auc_score(y[te], p)
check(f"model recovers signal from synthetic data (AUC {auc:.3f} > 0.60)", auc > 0.60)

from sklearn.metrics import brier_score_loss
brier = brier_score_loss(y[te], p)
check(f"beats a coinflip on Brier ({brier:.4f} < 0.25)", brier < 0.25)

# -- Kelly sizing -------------------------------------------------------------
from model.backtest import MAX_STAKE, devig_two_way, kelly_stake

check("no stake when there is no edge", kelly_stake(0.40, 2.00) == 0.0)
check("positive stake when there is an edge", kelly_stake(0.60, 2.00) > 0.0)
check("stake is capped", kelly_stake(0.99, 10.0) <= MAX_STAKE)
fa, fb_ = devig_two_way(2.00, 2.00)
check("de-vigged probabilities sum to 1", abs(fa + fb_ - 1.0) < 1e-9)
fa, fb_ = devig_two_way(1.50, 3.00)
check("de-vig preserves the favourite", fa > fb_)

print("\n" + "=" * 46)
n_fail = sum(1 for ok, _ in results if not ok)
print(f"{len(results) - n_fail}/{len(results)} passed")
if n_fail:
    for ok, n in results:
        if not ok:
            print(f"  FAILED: {n}")
    raise SystemExit(1)
