"""
Feature engineering.

Every function takes `as_of` and is strictly `match_date < as_of`. That single
discipline is what separates a real backtest from one that quietly sees the
future and reports a fake edge.

Feature ordering mirrors the analysis framework:
  1. current form and match log        (highest weight)
  2. opponent quality of those results
  3. surface-specific splits
  4. head-to-head, recency weighted
  5. rank -- as a validator only, not a primary signal

Closing odds are deliberately ABSENT. They live in a separate table and are used
only as a backtest benchmark. A model trained on market prices just learns to
reproduce the market.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import date, timedelta

FORM_N = 10
SEASON_DAYS = 365
STALE_DAYS = 21

RANK_TIERS = [(1, 10), (11, 25), (26, 50), (51, 100), (101, 250), (251, 99999)]
TIER_WEIGHT = [1.00, 0.80, 0.62, 0.45, 0.28, 0.15]   # beating a top-10 counts more


def _rows_before(conn: sqlite3.Connection, pid: str, as_of: str,
                 since: str | None = None, surface: str | None = None) -> list[dict]:
    sql = """SELECT match_date, surface, tour_level, retirement,
                    CASE WHEN winner_id = :pid THEN 1 ELSE 0 END AS won,
                    CASE WHEN winner_id = :pid THEN loser_rank ELSE winner_rank END AS opp_rank,
                    CASE WHEN winner_id = :pid THEN loser_id ELSE winner_id END AS opp_id
             FROM matches
             WHERE (winner_id = :pid OR loser_id = :pid) AND match_date < :as_of"""
    p = {"pid": pid, "as_of": as_of}
    if since:
        sql += " AND match_date >= :since"
        p["since"] = since
    if surface:
        sql += " AND surface = :surface"
        p["surface"] = surface
    sql += " ORDER BY match_date DESC"
    return [dict(r) for r in conn.execute(sql, p)]


def _tier_index(rank: int | None) -> int | None:
    if not rank:
        return None
    for i, (lo, hi) in enumerate(RANK_TIERS):
        if lo <= rank <= hi:
            return i
    return None


PRIOR_STRENGTH = 1.0   # Beta(1,1) pseudo-counts -- shrinks thin records to 0.5


def quality_score(rows: list[dict]) -> float:
    """
    Opponent-quality-adjusted win rate, shrunk toward 0.5.

    The naive version -- sum(weight * won) / sum(weight) -- looks right and is
    wrong: for an all-wins record the weight cancels, so beating a top-10 and
    beating a #300 both score exactly 1.0. The quality signal vanishes at
    precisely the moment it matters.

    Instead, treat each result as evidence added to a Beta prior:
      - a win over tier t adds  w_t  pseudo-wins   (beating a top-10 is worth more)
      - a loss to tier t adds  (1 - w_t + eps) pseudo-losses
        (losing to a top-10 barely hurts; losing to a #300 hurts a lot)

    A 5-0 record over #300s lands near 0.6; 5-0 over top-10s lands near 0.85.
    Thin records stay near 0.5, which is the honest answer for thin records.
    """
    pseudo_w = pseudo_l = 0.0
    for r in rows:
        ti = _tier_index(r["opp_rank"])
        w = TIER_WEIGHT[ti] if ti is not None else 0.10
        if r["won"]:
            pseudo_w += w
        else:
            pseudo_l += 1.0 - w + 0.15
    return (PRIOR_STRENGTH + pseudo_w) / (2 * PRIOR_STRENGTH + pseudo_w + pseudo_l)


def player_features(conn: sqlite3.Connection, pid: str, surface: str | None,
                    as_of: str) -> dict:
    """Single-player feature block. Returns staleness alongside the numbers."""
    season_start = (date.fromisoformat(as_of) - timedelta(days=SEASON_DAYS)).isoformat()
    recent = _rows_before(conn, pid, as_of)[:FORM_N]
    season = _rows_before(conn, pid, as_of, since=season_start)
    surf = _rows_before(conn, pid, as_of, since=season_start, surface=surface) if surface else []

    n_recent = len(recent)
    days_since = None
    if recent:
        days_since = (date.fromisoformat(as_of) - date.fromisoformat(recent[0]["match_date"])).days

    fortnight = (date.fromisoformat(as_of) - timedelta(days=14)).isoformat()
    load_14d = sum(1 for r in season if r["match_date"] >= fortnight)

    return {
        # 1. form
        "form_win_rate_10": sum(r["won"] for r in recent) / n_recent if n_recent else 0.5,
        "form_win_rate_5": (sum(r["won"] for r in recent[:5]) / min(5, n_recent)) if n_recent else 0.5,
        "n_recent": n_recent,
        "days_since_last": days_since if days_since is not None else 999,
        # 2. opponent quality
        "quality_score_season": quality_score(season),
        "quality_score_recent": quality_score(recent),
        "best_win_tier": min(
            (_tier_index(r["opp_rank"]) for r in season
             if r["won"] and _tier_index(r["opp_rank"]) is not None),
            default=len(RANK_TIERS),
        ),
        "worst_loss_tier": max(
            (_tier_index(r["opp_rank"]) for r in season
             if not r["won"] and _tier_index(r["opp_rank"]) is not None),
            default=0,
        ),
        # 3. surface
        "surface_win_rate": (sum(r["won"] for r in surf) / len(surf)) if surf else 0.5,
        "surface_n": len(surf),
        "surface_quality": quality_score(surf),
        # context
        "season_matches": len(season),
        "season_win_rate": (sum(r["won"] for r in season) / len(season)) if season else 0.5,
        "load_14d": load_14d,
        "retire_rate": (sum(r["retirement"] for r in season) / len(season)) if season else 0.0,
        # staleness -- carried through so serving can degrade confidence
        "_is_stale": days_since is None or days_since > STALE_DAYS,
        "_data_gap": n_recent == 0,
    }


def h2h_features(conn: sqlite3.Connection, a: str, b: str, as_of: str,
                 half_life_days: float = 540.0) -> dict:
    """Recency-weighted H2H. A 2019 meeting should not count like a 2026 one."""
    rows = conn.execute(
        """SELECT match_date, surface,
                  CASE WHEN winner_id = :a THEN 1 ELSE 0 END AS a_won
           FROM matches
           WHERE ((winner_id=:a AND loser_id=:b) OR (winner_id=:b AND loser_id=:a))
             AND match_date < :as_of""",
        {"a": a, "b": b, "as_of": as_of},
    ).fetchall()

    if not rows:
        return {"h2h_n": 0, "h2h_weighted": 0.5, "h2h_raw": 0.5}

    ref = date.fromisoformat(as_of)
    num = den = 0.0
    for r in rows:
        age = (ref - date.fromisoformat(r["match_date"])).days
        w = math.exp(-age / half_life_days)
        num += w * r["a_won"]
        den += w
    return {
        "h2h_n": len(rows),
        "h2h_weighted": num / den if den else 0.5,
        "h2h_raw": sum(r["a_won"] for r in rows) / len(rows),
    }


def rank_as_of(conn: sqlite3.Connection, pid: str, as_of: str) -> int | None:
    row = conn.execute(
        """SELECT rank FROM rankings
           WHERE player_id=? AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
           ORDER BY valid_from DESC LIMIT 1""",
        (pid, as_of, as_of),
    ).fetchone()
    return row["rank"] if row else None


def build_features(conn: sqlite3.Connection, p1: str, p2: str, surface: str | None,
                   as_of: str, tour_level: str | None = None) -> dict:
    """
    Full symmetric feature vector for (p1 vs p2).

    Emits DIFFERENCES (p1 minus p2) rather than raw pairs so the model can't
    learn "player listed first tends to win" -- there is no such signal, and a
    model that finds one is memorising your data ordering.
    """
    f1 = player_features(conn, p1, surface, as_of)
    f2 = player_features(conn, p2, surface, as_of)
    h = h2h_features(conn, p1, p2, as_of)

    r1, r2 = rank_as_of(conn, p1, as_of), rank_as_of(conn, p2, as_of)
    lr1 = math.log(r1) if r1 else math.log(1500)
    lr2 = math.log(r2) if r2 else math.log(1500)

    feats = {
        "d_form_10": f1["form_win_rate_10"] - f2["form_win_rate_10"],
        "d_form_5": f1["form_win_rate_5"] - f2["form_win_rate_5"],
        "d_quality_season": f1["quality_score_season"] - f2["quality_score_season"],
        "d_quality_recent": f1["quality_score_recent"] - f2["quality_score_recent"],
        "d_best_win_tier": f2["best_win_tier"] - f1["best_win_tier"],   # lower is better
        "d_worst_loss_tier": f2["worst_loss_tier"] - f1["worst_loss_tier"],
        "d_surface_wr": f1["surface_win_rate"] - f2["surface_win_rate"],
        "d_surface_quality": f1["surface_quality"] - f2["surface_quality"],
        "d_season_wr": f1["season_win_rate"] - f2["season_win_rate"],
        "d_log_rank": lr2 - lr1,                                        # positive => p1 better
        "d_days_since": f2["days_since_last"] - f1["days_since_last"],
        "d_load_14d": f1["load_14d"] - f2["load_14d"],
        "d_retire_rate": f2["retire_rate"] - f1["retire_rate"],
        "h2h_weighted": h["h2h_weighted"],
        "h2h_n": h["h2h_n"],
        "min_surface_n": min(f1["surface_n"], f2["surface_n"]),
        "min_season_matches": min(f1["season_matches"], f2["season_matches"]),
        "is_challenger": 1 if tour_level == "C" else 0,
    }
    meta = {
        "p1_stale": f1["_is_stale"], "p2_stale": f2["_is_stale"],
        "p1_gap": f1["_data_gap"], "p2_gap": f2["_data_gap"],
        "p1_days_since": f1["days_since_last"], "p2_days_since": f2["days_since_last"],
        "p1_rank": r1, "p2_rank": r2,
    }
    return {"features": feats, "meta": meta}


FEATURE_NAMES = [
    "d_form_10", "d_form_5", "d_quality_season", "d_quality_recent",
    "d_best_win_tier", "d_worst_loss_tier", "d_surface_wr", "d_surface_quality",
    "d_season_wr", "d_log_rank", "d_days_since", "d_load_14d", "d_retire_rate",
    "h2h_weighted", "h2h_n", "min_surface_n", "min_season_matches", "is_challenger",
]
