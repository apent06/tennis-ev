"""
Favorite-longshot bias test.

Question: does the market systematically misprice favorites or underdogs?

Method: for every match with closing odds, remove the vig to get the market's
fair probability for each side. Bucket every side by that probability, then
check how often sides in each bucket actually won.

If the market is efficient, observed win rate should match implied probability
in every bucket. Systematic gaps = mispricing you could bet against.

Note this tests the market, not the model. No model involved at all.

Run: python bias_test.py
"""

import sqlite3
from collections import defaultdict

import os
import sys

# Resolve paths from the project root, not the caller's working directory, so
# this runs the same whether invoked as `python analysis/x.py` or from inside
# the folder.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DB = os.path.join(ROOT, "tennis.db")



def devig(odds_w, odds_l):
    """Remove bookmaker margin so the two sides sum to 1.0."""
    iw, il = 1 / odds_w, 1 / odds_l
    total = iw + il
    return iw / total, il / total


def main():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT m.tour, m.tour_level, m.match_date,
               o.winner_odds, o.loser_odds
        FROM matches m
        JOIN closing_odds o ON o.match_key = m.match_key
        WHERE o.winner_odds > 1.0 AND o.loser_odds > 1.0
    """).fetchall()

    print(f"matches with closing odds: {len(rows)}\n")

    # Each match contributes two observations: the winner's side (won=1) and
    # the loser's side (won=0). Otherwise you only ever see winning sides.
    edges = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
             0.60, 0.70, 0.80, 0.90, 0.95, 1.0]

    buckets = defaultdict(lambda: {"n": 0, "wins": 0, "implied_sum": 0.0,
                                   "staked": 0.0, "returned": 0.0})
    by_level = defaultdict(lambda: defaultdict(
        lambda: {"n": 0, "wins": 0, "implied_sum": 0.0}))

    def bucket_of(p):
        for i in range(len(edges) - 1):
            if edges[i] <= p < edges[i + 1]:
                return i
        return len(edges) - 2

    for r in rows:
        fair_w, fair_l = devig(r["winner_odds"], r["loser_odds"])
        # (fair probability, actually won, price you'd have gotten)
        for fair, won, price in ((fair_w, 1, r["winner_odds"]),
                                 (fair_l, 0, r["loser_odds"])):
            b = bucket_of(fair)
            buckets[b]["n"] += 1
            buckets[b]["wins"] += won
            buckets[b]["implied_sum"] += fair
            # flat 1-unit bet on every side in this bucket, at the real price
            buckets[b]["staked"] += 1.0
            buckets[b]["returned"] += price if won else 0.0

            lv = by_level[r["tour_level"]][b]
            lv["n"] += 1
            lv["wins"] += won
            lv["implied_sum"] += fair

    print("=" * 78)
    print("MARKET CALIBRATION (all matches)")
    print("=" * 78)
    print(f"{'bucket':>12} {'n':>7} {'implied':>9} {'actual':>9} "
          f"{'gap':>8} {'flat ROI':>10}")
    print("-" * 78)

    for b in sorted(buckets):
        d = buckets[b]
        if d["n"] < 30:
            continue
        implied = d["implied_sum"] / d["n"]
        actual = d["wins"] / d["n"]
        roi = (d["returned"] - d["staked"]) / d["staked"]
        label = f"{edges[b]:.2f}-{edges[b+1]:.2f}"
        flag = ""
        if abs(actual - implied) > 0.02:
            flag = "  <-- underpriced" if actual > implied else "  <-- overpriced"
        print(f"{label:>12} {d['n']:>7} {implied:>9.3f} {actual:>9.3f} "
              f"{actual-implied:>+8.3f} {roi:>+9.2%}{flag}")

    print()
    print("Reading this: 'implied' is what the market said would happen after")
    print("removing vig. 'actual' is what happened. A positive gap means that")
    print("bucket won MORE often than priced (underpriced). 'flat ROI' is what")
    print("you'd have made betting one unit on every side in that bucket at the")
    print("actual odds offered, vig included.")
    print()

    # favorites vs underdogs, simple split
    fav_n = fav_w = fav_imp = 0
    dog_n = dog_w = dog_imp = 0
    fav_stake = fav_ret = dog_stake = dog_ret = 0.0
    for r in rows:
        fair_w, fair_l = devig(r["winner_odds"], r["loser_odds"])
        for fair, won, price in ((fair_w, 1, r["winner_odds"]),
                                 (fair_l, 0, r["loser_odds"])):
            if fair >= 0.5:
                fav_n += 1; fav_w += won; fav_imp += fair
                fav_stake += 1.0; fav_ret += price if won else 0.0
            else:
                dog_n += 1; dog_w += won; dog_imp += fair
                dog_stake += 1.0; dog_ret += price if won else 0.0

    print("=" * 78)
    print("FAVORITES vs UNDERDOGS")
    print("=" * 78)
    print(f"{'side':>12} {'n':>7} {'implied':>9} {'actual':>9} {'gap':>8} {'flat ROI':>10}")
    print("-" * 78)
    print(f"{'favorites':>12} {fav_n:>7} {fav_imp/fav_n:>9.3f} {fav_w/fav_n:>9.3f} "
          f"{fav_w/fav_n - fav_imp/fav_n:>+8.3f} {(fav_ret-fav_stake)/fav_stake:>+9.2%}")
    print(f"{'underdogs':>12} {dog_n:>7} {dog_imp/dog_n:>9.3f} {dog_w/dog_n:>9.3f} "
          f"{dog_w/dog_n - dog_imp/dog_n:>+8.3f} {(dog_ret-dog_stake)/dog_stake:>+9.2%}")
    print()

    print("=" * 78)
    print("BY TOUR LEVEL (favorites only, implied >= 0.5)")
    print("=" * 78)
    print(f"{'level':>8} {'n':>7} {'implied':>9} {'actual':>9} {'gap':>8}")
    print("-" * 78)
    for level in sorted(by_level):
        n = w = 0
        imp = 0.0
        for b, d in by_level[level].items():
            if edges[b] >= 0.5:
                n += d["n"]; w += d["wins"]; imp += d["implied_sum"]
        if n < 100:
            continue
        print(f"{level:>8} {n:>7} {imp/n:>9.3f} {w/n:>9.3f} {w/n - imp/n:>+8.3f}")

    print()
    print("Caveat: a gap needs to be both large and consistent across buckets to")
    print("be a real effect rather than noise. With n in the hundreds, gaps under")
    print("about 0.02 are not meaningful.")


if __name__ == "__main__":
    main()
