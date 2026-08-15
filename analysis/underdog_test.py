"""
Subset test: do underdogs with better fundamentals beat their market price?

The hypothesis: the market anchors on ranking and reputation. Sometimes the
player priced as an underdog actually has better recent form and has been
beating better opponents. In those specific matches, the underdog should win
more often than the price implies.

This tests it directly against real data. No model involved, it compares raw
form/quality features against the market's own probability.

Method:
  1. For every match with closing odds, identify which side the market made the
     underdog (de-vigged probability < 0.5).
  2. Compute both players' fundamentals as of the day before the match:
       - last-10 win rate
       - opponent-quality score (wins weighted by opponent rank tier)
       - surface win rate over the past 365 days
  3. Filter to matches where the underdog is better on those measures.
  4. Check how often those underdogs actually won vs what the market implied,
     and what flat-staking them would have returned.

Run: python underdog_test.py
"""

import sqlite3
from collections import defaultdict
from datetime import date, timedelta

import os
import sys

# Resolve paths from the project root, not the caller's working directory, so
# this runs the same whether invoked as `python analysis/x.py` or from inside
# the folder.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DB = os.path.join(ROOT, "tennis.db")

from model.features import player_features  # noqa: E402
MIN_MATCHES = 8          # need some history for fundamentals to mean anything
MIN_SUBSET = 50          # below this, don't report a bucket


def devig(odds_w, odds_l):
    iw, il = 1 / odds_w, 1 / odds_l
    t = iw + il
    return iw / t, il / t


def summarize(name, rows):
    """rows = list of (implied_prob, won, price)"""
    n = len(rows)
    if n < MIN_SUBSET:
        print(f"{name:<44} n={n:<6} (too few to report)")
        return
    implied = sum(r[0] for r in rows) / n
    actual = sum(r[1] for r in rows) / n
    staked = float(n)
    returned = sum(r[2] for r in rows if r[1])
    roi = (returned - staked) / staked
    gap = actual - implied
    flag = ""
    if abs(gap) > 0.02:
        flag = "  <-- WON MORE THAN PRICED" if gap > 0 else "  <-- won less than priced"
    print(f"{name:<44} n={n:<6} implied={implied:.3f} actual={actual:.3f} "
          f"gap={gap:+.3f} ROI={roi:+7.2%}{flag}")


def main():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT m.match_key, m.match_date, m.surface, m.tour_level, m.tour,
               m.winner_id, m.loser_id, o.winner_odds, o.loser_odds
        FROM matches m
        JOIN closing_odds o ON o.match_key = m.match_key
        WHERE o.winner_odds > 1.0 AND o.loser_odds > 1.0
          AND m.winner_id IS NOT NULL AND m.loser_id IS NOT NULL
          AND m.match_date >= '2023-01-01'
        ORDER BY m.match_date
    """).fetchall()

    print(f"matches with odds and player ids: {len(rows)}")
    print("computing fundamentals (this takes a few minutes)...\n")

    # buckets of (implied, won, price) for the underdog side only
    all_dogs = []
    better_form = []
    better_quality = []
    better_surface = []
    better_all_three = []
    better_two_plus = []
    by_level = defaultdict(list)
    worse_all_three = []

    skipped = 0
    for i, r in enumerate(rows):
        if i % 2000 == 0 and i:
            print(f"  {i}/{len(rows)}")

        fair_w, fair_l = devig(r["winner_odds"], r["loser_odds"])

        # identify the underdog side and whether it won
        if fair_w < fair_l:
            dog_id, fav_id = r["winner_id"], r["loser_id"]
            dog_implied, dog_price, dog_won = fair_w, r["winner_odds"], 1
        elif fair_l < fair_w:
            dog_id, fav_id = r["loser_id"], r["winner_id"]
            dog_implied, dog_price, dog_won = fair_l, r["loser_odds"], 0
        else:
            continue  # pick'em, no underdog

        as_of = r["match_date"]
        f_dog = player_features(conn, dog_id, r["surface"], as_of)
        f_fav = player_features(conn, fav_id, r["surface"], as_of)

        if (f_dog["season_matches"] < MIN_MATCHES
                or f_fav["season_matches"] < MIN_MATCHES):
            skipped += 1
            continue

        obs = (dog_implied, dog_won, dog_price)
        all_dogs.append(obs)
        by_level[r["tour_level"]].append(obs)

        form_better = f_dog["form_win_rate_10"] > f_fav["form_win_rate_10"]
        qual_better = f_dog["quality_score_season"] > f_fav["quality_score_season"]
        surf_better = (f_dog["surface_n"] >= 3 and f_fav["surface_n"] >= 3
                       and f_dog["surface_win_rate"] > f_fav["surface_win_rate"])

        if form_better:
            better_form.append(obs)
        if qual_better:
            better_quality.append(obs)
        if surf_better:
            better_surface.append(obs)
        n_better = sum([form_better, qual_better, surf_better])
        if n_better >= 2:
            better_two_plus.append(obs)
        if n_better == 3:
            better_all_three.append(obs)
        if n_better == 0:
            worse_all_three.append(obs)

    print(f"\nskipped {skipped} matches (insufficient history)\n")

    print("=" * 100)
    print("UNDERDOGS, SPLIT BY WHETHER THEIR FUNDAMENTALS BEAT THE FAVORITE'S")
    print("=" * 100)
    summarize("ALL underdogs (baseline)", all_dogs)
    print()
    summarize("underdog has better last-10 form", better_form)
    summarize("underdog has better opponent quality", better_quality)
    summarize("underdog has better surface record", better_surface)
    print()
    summarize("underdog better on 2+ of the three", better_two_plus)
    summarize("underdog better on ALL THREE", better_all_three)
    summarize("underdog worse on all three (control)", worse_all_three)

    print()
    print("=" * 100)
    print("ALL UNDERDOGS BY TOUR LEVEL")
    print("=" * 100)
    for lvl in sorted(by_level):
        summarize(f"tour level {lvl}", by_level[lvl])

    print()
    print("How to read this:")
    print("  'implied' = what the market priced the underdog at, vig removed.")
    print("  'actual'  = how often those underdogs actually won.")
    print("  A positive gap means the market underpriced that group.")
    print("  ROI is flat 1-unit stakes at the real odds, vig included.")
    print()
    print("The control row matters: if underdogs with BETTER fundamentals and")
    print("underdogs with WORSE fundamentals both land near zero gap, the market")
    print("has already priced fundamentals in and there is nothing here.")
    print("A real effect means the 'all three' row is clearly positive AND the")
    print("control row is clearly negative.")


if __name__ == "__main__":
    main()
