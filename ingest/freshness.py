"""
Freshness measurement + the feature queries that read off the match table.

Two jobs here:

1. Turn "the data feels stale" into a number: observed ingestion lag per tour
   level, and an SLO check that alerts when the newest match is too old.

2. Serve features WITH their staleness attached, so the API can degrade
   confidence instead of quietly returning a number built on three-week-old
   form.
"""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta

# Lag budgets in days. Challengers are usually the worst offender -- measure
# yours and tune these rather than trusting the defaults.
SLO_DAYS = {"G": 2, "M": 2, "A": 3, "C": 5, "S": 7}
DEFAULT_SLO = 5
STALE_FORM_DAYS = 21   # if a player's newest match is older than this, flag it


def observed_lag(conn: sqlite3.Connection, days_back: int = 90) -> list[dict]:
    """Median/p90 lag between match date and when we first saw the row."""
    cutoff = (date.today() - timedelta(days=days_back)).isoformat()
    rows = conn.execute(
        """SELECT tour_level, match_date, first_seen_at
           FROM matches WHERE match_date >= ?""",
        (cutoff,),
    ).fetchall()

    buckets: dict[str, list[float]] = {}
    for r in rows:
        try:
            seen = datetime.fromisoformat(r["first_seen_at"]).date()
            md = date.fromisoformat(r["match_date"])
        except (ValueError, TypeError):
            continue
        buckets.setdefault(r["tour_level"] or "?", []).append((seen - md).days)

    out = []
    for level, lags in sorted(buckets.items()):
        lags.sort()
        n = len(lags)
        out.append({
            "tour_level": level,
            "n": n,
            "median_lag_days": lags[n // 2],
            "p90_lag_days": lags[min(int(n * 0.9), n - 1)],
            "max_lag_days": lags[-1],
        })
    return out


def check_slo(conn: sqlite3.Connection) -> list[dict]:
    """Returns breaches. Empty list == healthy. Wire this to an alert."""
    today = date.today()
    breaches = []
    for level, budget in SLO_DAYS.items():
        row = conn.execute(
            "SELECT MAX(match_date) AS newest, COUNT(*) AS n FROM matches WHERE tour_level = ?",
            (level,),
        ).fetchone()
        if not row or not row["newest"]:
            continue
        age = (today - date.fromisoformat(row["newest"])).days
        if age > budget:
            breaches.append({
                "tour_level": level, "newest_match": row["newest"],
                "age_days": age, "budget_days": budget,
            })
    return breaches


# ---------------------------------------------------------------- features --

def player_form(conn: sqlite3.Connection, player_id: str, n: int = 10,
                as_of: str | None = None) -> dict:
    """
    Last-n results with opponent quality attached.

    `as_of` is mandatory discipline for backtests: pass the match date so you
    only ever see what was knowable before the match.
    """
    as_of = as_of or date.today().isoformat()
    rows = conn.execute(
        """SELECT match_date, tournament, tour_level, surface, score,
                  CASE WHEN winner_id = :pid THEN 1 ELSE 0 END AS won,
                  CASE WHEN winner_id = :pid THEN loser_name ELSE winner_name END AS opponent,
                  CASE WHEN winner_id = :pid THEN loser_rank ELSE winner_rank END AS opponent_rank
           FROM matches
           WHERE (winner_id = :pid OR loser_id = :pid) AND match_date < :as_of
           ORDER BY match_date DESC LIMIT :n""",
        {"pid": player_id, "as_of": as_of, "n": n},
    ).fetchall()

    results = [dict(r) for r in rows]
    newest = results[0]["match_date"] if results else None
    days_since = (
        (date.fromisoformat(as_of) - date.fromisoformat(newest)).days if newest else None
    )
    return {
        "player_id": player_id,
        "matches": results,
        "form_string": "".join("W" if r["won"] else "L" for r in results),
        "wins": sum(r["won"] for r in results),
        "losses": sum(1 - r["won"] for r in results),
        "newest_match_date": newest,
        "days_since_last_match": days_since,
        "is_stale": days_since is None or days_since > STALE_FORM_DAYS,
        "data_gap": len(results) == 0,
    }


def surface_splits(conn: sqlite3.Connection, player_id: str, season: int | None = None,
                   as_of: str | None = None) -> dict:
    as_of = as_of or date.today().isoformat()
    sql = """SELECT surface,
                    SUM(CASE WHEN winner_id = :pid THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN loser_id  = :pid THEN 1 ELSE 0 END) AS losses
             FROM matches
             WHERE (winner_id = :pid OR loser_id = :pid)
               AND match_date < :as_of AND surface IS NOT NULL"""
    params = {"pid": player_id, "as_of": as_of}
    if season:
        sql += " AND match_date >= :y0 AND match_date <= :y1"
        params |= {"y0": f"{season}-01-01", "y1": f"{season}-12-31"}
    sql += " GROUP BY surface"
    return {r["surface"]: {"wins": r["wins"], "losses": r["losses"]}
            for r in conn.execute(sql, params)}


def quality_wins(conn: sqlite3.Connection, player_id: str, window_days: int = 365,
                 as_of: str | None = None) -> dict:
    """
    Win rate bucketed by opponent rank tier -- the opponent-quality view.

    Uses the rank stored ON the match row, so it's point-in-time correct with
    no rankings join and no leakage.
    """
    as_of = as_of or date.today().isoformat()
    start = (date.fromisoformat(as_of) - timedelta(days=window_days)).isoformat()
    rows = conn.execute(
        """SELECT CASE WHEN winner_id = :pid THEN 1 ELSE 0 END AS won,
                  CASE WHEN winner_id = :pid THEN loser_rank ELSE winner_rank END AS opp_rank
           FROM matches
           WHERE (winner_id = :pid OR loser_id = :pid)
             AND match_date >= :start AND match_date < :as_of""",
        {"pid": player_id, "start": start, "as_of": as_of},
    ).fetchall()

    tiers = {"top10": (1, 10), "11_25": (11, 25), "26_50": (26, 50),
             "51_100": (51, 100), "101_250": (101, 250), "250_plus": (251, 99999)}
    out = {t: {"wins": 0, "losses": 0} for t in tiers}
    out["unranked_opponent"] = {"wins": 0, "losses": 0}

    for r in rows:
        rank = r["opp_rank"]
        bucket = "unranked_opponent"
        if rank:
            for t, (lo, hi) in tiers.items():
                if lo <= rank <= hi:
                    bucket = t
                    break
        out[bucket]["wins" if r["won"] else "losses"] += 1
    return out


def head_to_head(conn: sqlite3.Connection, a: str, b: str,
                 as_of: str | None = None) -> list[dict]:
    as_of = as_of or date.today().isoformat()
    return [dict(r) for r in conn.execute(
        """SELECT match_date, tournament, surface, score,
                  CASE WHEN winner_id = :a THEN :a ELSE :b END AS winner_id
           FROM matches
           WHERE ((winner_id = :a AND loser_id = :b) OR (winner_id = :b AND loser_id = :a))
             AND match_date < :as_of
           ORDER BY match_date DESC""",
        {"a": a, "b": b, "as_of": as_of},
    )]
