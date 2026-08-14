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

# Staleness is graduated, not binary. A three-week gap between tournaments is
# routine on tour; a three-month gap usually means injury or a layoff, and form
# recorded before a long absence often doesn't carry over. Treating those the
# same throws away the distinction that matters.
STALE_SOFT_DAYS = 21    # note it, don't alarm
STALE_HARD_DAYS = 45    # genuinely questionable
# How stretched the last-N window is. Ten matches over six weeks is a normal
# run of tournaments; ten matches spread over eight months is not "recent form"
# in any useful sense, even if the most recent one was yesterday.
WINDOW_STRETCHED_DAYS = 180

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


def dominance(conn: sqlite3.Connection, pid: str, as_of: str,
              window_days: int = SEASON_DAYS) -> dict:
    """
    How comfortably a player has been winning and losing.

    Match results are one bit each; game counts are the closest thing this data
    has to a margin. A player scraping 7-6 7-6 wins is a different proposition
    from one winning 6-2 6-1, and win/loss alone cannot tell them apart.

    Serve/return stats would be strictly better here. They aren't in the
    source, so games-won share is the available proxy.
    """
    start = (date.fromisoformat(as_of) - timedelta(days=window_days)).isoformat()
    rows = conn.execute(
        """SELECT CASE WHEN winner_id = :pid THEN 1 ELSE 0 END AS won,
                  w_games, l_games, w_sets, l_sets, best_of
           FROM matches
           WHERE (winner_id = :pid OR loser_id = :pid)
             AND match_date >= :start AND match_date < :as_of
             AND w_games IS NOT NULL AND l_games IS NOT NULL""",
        {"pid": pid, "start": start, "as_of": as_of},
    ).fetchall()

    if not rows:
        return {"game_share": 0.5, "set_share": 0.5, "n_scored": 0,
                "close_rate": 0.0}

    mine = theirs = 0
    sets_mine = sets_theirs = 0
    close = 0
    for r in rows:
        wg, lg = r["w_games"] or 0, r["l_games"] or 0
        ws, ls = r["w_sets"] or 0, r["l_sets"] or 0
        if r["won"]:
            mine += wg; theirs += lg
            sets_mine += ws; sets_theirs += ls
        else:
            mine += lg; theirs += wg
            sets_mine += ls; sets_theirs += ws
        total = wg + lg
        if total and abs(wg - lg) / total < 0.12:      # decided by a hair
            close += 1

    tg = mine + theirs
    ts = sets_mine + sets_theirs
    return {
        "game_share": mine / tg if tg else 0.5,
        "set_share": sets_mine / ts if ts else 0.5,
        "n_scored": len(rows),
        "close_rate": close / len(rows),
    }


def serve_profile(conn: sqlite3.Connection, pid: str, as_of: str,
                  window_days: int = SEASON_DAYS) -> dict:
    """
    Serve and return rates, when the source provides them.

    Returns available=False when the columns are empty, which is the case for
    tennis-data.co.uk. Downstream emits neutral values and a flag rather than
    zeros -- a player with no serve data is not a player who wins no service
    points, and conflating those would be a serious feature bug.
    """
    start = (date.fromisoformat(as_of) - timedelta(days=window_days)).isoformat()
    r = conn.execute(
        """SELECT
             SUM(CASE WHEN winner_id = :pid THEN w_svpt    ELSE l_svpt    END) AS svpt,
             SUM(CASE WHEN winner_id = :pid THEN w_1st_in  ELSE l_1st_in  END) AS first_in,
             SUM(CASE WHEN winner_id = :pid THEN w_1st_won ELSE l_1st_won END) AS first_won,
             SUM(CASE WHEN winner_id = :pid THEN w_2nd_won ELSE l_2nd_won END) AS second_won,
             SUM(CASE WHEN winner_id = :pid THEN w_ace     ELSE l_ace     END) AS ace,
             SUM(CASE WHEN winner_id = :pid THEN w_df      ELSE l_df      END) AS df,
             SUM(CASE WHEN winner_id = :pid THEN w_bp_saved ELSE l_bp_saved END) AS bp_saved,
             SUM(CASE WHEN winner_id = :pid THEN w_bp_faced ELSE l_bp_faced END) AS bp_faced
           FROM matches
           WHERE (winner_id = :pid OR loser_id = :pid)
             AND match_date >= :start AND match_date < :as_of""",
        {"pid": pid, "start": start, "as_of": as_of},
    ).fetchone()

    svpt = r["svpt"] or 0
    if not svpt:
        return {"available": False, "serve_won": 0.5, "ace_rate": 0.0,
                "df_rate": 0.0, "bp_save_rate": 0.5, "first_in_rate": 0.5}

    won = (r["first_won"] or 0) + (r["second_won"] or 0)
    faced = r["bp_faced"] or 0
    return {
        "available": True,
        "serve_won": won / svpt,
        "ace_rate": (r["ace"] or 0) / svpt,
        "df_rate": (r["df"] or 0) / svpt,
        "bp_save_rate": (r["bp_saved"] or 0) / faced if faced else 0.5,
        "first_in_rate": (r["first_in"] or 0) / svpt,
    }


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

    # How much calendar time the last-N matches actually cover.
    window_days = None
    if len(recent) >= 2:
        window_days = (date.fromisoformat(recent[0]["match_date"])
                       - date.fromisoformat(recent[-1]["match_date"])).days

    if days_since is None:
        tier = "none"
    elif days_since > STALE_HARD_DAYS:
        tier = "hard"
    elif days_since > STALE_SOFT_DAYS:
        tier = "soft"
    else:
        tier = "fresh"

    return {
        # 1. form
        "form_win_rate_10": sum(r["won"] for r in recent) / n_recent if n_recent else 0.5,
        "wins_10": sum(r["won"] for r in recent),
        "losses_10": n_recent - sum(r["won"] for r in recent),
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
        "_stale_tier": tier,
        "_is_stale": tier in ("hard", "none"),
        "_form_window_days": window_days,
        "_window_stretched": window_days is not None and window_days > WINDOW_STRETCHED_DAYS,
        "_oldest_form_match": recent[-1]["match_date"] if recent else None,
        "_newest_form_match": recent[0]["match_date"] if recent else None,
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
    """
    Ranking as it stood before `as_of`.

    Prefers the versioned rankings table. Falls back to the rank recorded ON
    the player's most recent match row, because tennis-data.co.uk stores rank
    per match rather than as weekly snapshots -- with that source the rankings
    table is empty and the primary lookup returns None for everyone.

    That failure is silent and severe: both players fall back to the same
    default, d_log_rank becomes exactly zero on every match, and the ranking
    feature quietly stops existing. The baselines harness caught it by scoring
    rank-only at precisely 0.500 AUC.
    """
    row = conn.execute(
        """SELECT rank FROM rankings
           WHERE player_id=? AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
           ORDER BY valid_from DESC LIMIT 1""",
        (pid, as_of, as_of),
    ).fetchone()
    if row and row["rank"]:
        return int(row["rank"])

    row = conn.execute(
        """SELECT CASE WHEN winner_id = :pid THEN winner_rank
                       ELSE loser_rank END AS rk
           FROM matches
           WHERE (winner_id = :pid OR loser_id = :pid) AND match_date < :as_of
             AND (CASE WHEN winner_id = :pid THEN winner_rank
                       ELSE loser_rank END) IS NOT NULL
           ORDER BY match_date DESC, rowid DESC LIMIT 1""",
        {"pid": pid, "as_of": as_of},
    ).fetchone()
    return int(row["rk"]) if row and row["rk"] else None


def build_features(conn: sqlite3.Connection, p1: str, p2: str, surface: str | None,
                   as_of: str, tour_level: str | None = None,
                   best_of: int | None = None, court: str | None = None) -> dict:
    """
    Full symmetric feature vector for (p1 vs p2).

    Emits DIFFERENCES (p1 minus p2) rather than raw pairs so the model can't
    learn "player listed first tends to win" -- there is no such signal, and a
    model that finds one is memorising your data ordering.
    """
    from .elo import BASE_RATING, as_of as elo_as_of, win_probability

    f1 = player_features(conn, p1, surface, as_of)
    f2 = player_features(conn, p2, surface, as_of)
    h = h2h_features(conn, p1, p2, as_of)
    d1 = dominance(conn, p1, as_of)
    d2 = dominance(conn, p2, as_of)
    s1 = serve_profile(conn, p1, as_of)
    s2 = serve_profile(conn, p2, as_of)

    e1, n1 = elo_as_of(conn, p1, as_of)
    e2, n2 = elo_as_of(conn, p2, as_of)
    es1, _ = elo_as_of(conn, p1, as_of, surface) if surface else (BASE_RATING, 0)
    es2, _ = elo_as_of(conn, p2, as_of, surface) if surface else (BASE_RATING, 0)

    serve_ok = s1["available"] and s2["available"]

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

        # --- Elo: recursive opponent quality, no hand-set tier weights ---
        "d_elo": e1 - e2,
        "d_elo_surface": es1 - es2,
        "elo_prob": win_probability(e1, e2),
        "elo_surface_prob": win_probability(es1, es2),
        "min_elo_matches": min(n1, n2),

        # --- how comfortably they have been winning ---
        "d_game_share": d1["game_share"] - d2["game_share"],
        "d_set_share": d1["set_share"] - d2["set_share"],
        "d_close_rate": d1["close_rate"] - d2["close_rate"],
        "min_scored_matches": min(d1["n_scored"], d2["n_scored"]),

        # --- match shape ---
        "is_bo5": 1 if best_of == 5 else 0,
        "is_indoor": 1 if (court or "").lower().startswith("in") else 0,

        # --- serve/return. Neutral zeros when unavailable, with a flag so the
        #     model can tell 'no data' from 'genuinely average'. ---
        "serve_available": 1 if serve_ok else 0,
        "d_serve_won": (s1["serve_won"] - s2["serve_won"]) if serve_ok else 0.0,
        "d_ace_rate": (s1["ace_rate"] - s2["ace_rate"]) if serve_ok else 0.0,
        "d_df_rate": (s2["df_rate"] - s1["df_rate"]) if serve_ok else 0.0,
        "d_bp_save": (s1["bp_save_rate"] - s2["bp_save_rate"]) if serve_ok else 0.0,
        "d_first_in": (s1["first_in_rate"] - s2["first_in_rate"]) if serve_ok else 0.0,
    }
    meta = {
        "p1_stale_tier": f1["_stale_tier"], "p2_stale_tier": f2["_stale_tier"],
        "p1_stale": f1["_is_stale"], "p2_stale": f2["_is_stale"],
        "p1_gap": f1["_data_gap"], "p2_gap": f2["_data_gap"],
        "p1_days_since": f1["days_since_last"], "p2_days_since": f2["days_since_last"],
        "p1_window_days": f1["_form_window_days"], "p2_window_days": f2["_form_window_days"],
        "p1_window_stretched": f1["_window_stretched"],
        "p2_window_stretched": f2["_window_stretched"],
        "p1_form_from": f1["_oldest_form_match"], "p1_form_to": f1["_newest_form_match"],
        "p2_form_from": f2["_oldest_form_match"], "p2_form_to": f2["_newest_form_match"],
        "p1_rank": r1, "p2_rank": r2,
        "p1_elo": round(e1), "p2_elo": round(e2),
        "p1_elo_surface": round(es1), "p2_elo_surface": round(es2),
        "serve_stats_available": serve_ok,
    }
    return {"features": feats, "meta": meta}


FEATURE_NAMES = [
    "d_form_10", "d_form_5", "d_quality_season", "d_quality_recent",
    "d_best_win_tier", "d_worst_loss_tier", "d_surface_wr", "d_surface_quality",
    "d_season_wr", "d_log_rank", "d_days_since", "d_load_14d", "d_retire_rate",
    "h2h_weighted", "h2h_n", "min_surface_n", "min_season_matches", "is_challenger",
    "d_elo", "d_elo_surface", "elo_prob", "elo_surface_prob", "min_elo_matches",
    "d_game_share", "d_set_share", "d_close_rate", "min_scored_matches",
    "is_bo5", "is_indoor",
    "serve_available", "d_serve_won", "d_ace_rate", "d_df_rate",
    "d_bp_save", "d_first_in",
]

# Value each feature takes when the two players are indistinguishable on it.
# Used to attribute a prediction: neutralise one feature, re-predict, and the
# movement is what that feature was worth.
#
# Context features (sample sizes, tour level) have no neutral -- they describe
# how much evidence there is, not who is favoured -- so they're excluded from
# attribution rather than given a fake baseline.
NEUTRAL = {
    "d_form_10": 0.0, "d_form_5": 0.0,
    "d_quality_season": 0.0, "d_quality_recent": 0.0,
    "d_best_win_tier": 0.0, "d_worst_loss_tier": 0.0,
    "d_surface_wr": 0.0, "d_surface_quality": 0.0,
    "d_season_wr": 0.0, "d_log_rank": 0.0,
    "d_days_since": 0.0, "d_load_14d": 0.0, "d_retire_rate": 0.0,
    "h2h_weighted": 0.5,
    "d_elo": 0.0, "d_elo_surface": 0.0,
    "elo_prob": 0.5, "elo_surface_prob": 0.5,
    "d_game_share": 0.0, "d_set_share": 0.0, "d_close_rate": 0.0,
    "d_serve_won": 0.0, "d_ace_rate": 0.0, "d_df_rate": 0.0,
    "d_bp_save": 0.0, "d_first_in": 0.0,
}

# Plain-language names for the interface.
FEATURE_LABELS = {
    "d_form_10": "Form over last 10",
    "d_form_5": "Form over last 5",
    "d_quality_season": "Quality of opponents beaten (season)",
    "d_quality_recent": "Quality of opponents beaten (recent)",
    "d_best_win_tier": "Best win of the season",
    "d_worst_loss_tier": "Worst loss of the season",
    "d_surface_wr": "Record on this surface",
    "d_surface_quality": "Surface record vs quality opponents",
    "d_season_wr": "Season win rate",
    "d_log_rank": "Ranking gap",
    "d_days_since": "Rest since last match",
    "d_load_14d": "Matches in last fortnight",
    "d_retire_rate": "Retirement rate",
    "h2h_weighted": "Head-to-head (recent meetings weighted)",
    "d_elo": "Elo rating gap",
    "d_elo_surface": "Elo gap on this surface",
    "elo_prob": "Elo's own estimate",
    "elo_surface_prob": "Elo's estimate on this surface",
    "d_game_share": "Share of games won",
    "d_set_share": "Share of sets won",
    "d_close_rate": "How often matches go to the wire",
    "d_serve_won": "Service points won",
    "d_ace_rate": "Ace rate",
    "d_df_rate": "Double fault rate",
    "d_bp_save": "Break points saved",
    "d_first_in": "First serves in",
}
