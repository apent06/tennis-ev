"""
Elo ratings for tennis, computed by chronological replay.

Why Elo alongside the feature model: it captures opponent quality recursively
in a way hand-built tier weights can't. Beating a player who has been beating
strong players raises your rating more, automatically, without anyone deciding
what a "quality win" is worth. In tennis it's a stubbornly good baseline -- if
an eighteen-feature model can't beat surface Elo, the features aren't earning
their keep.

Leak-free by construction: matches are replayed in date order, and each
player's rating BEFORE the match is written onto that match row. Nothing ever
sees a rating that incorporates the result it's trying to predict.

Two ratings per player:
  - overall, updated on every match
  - surface-specific, updated only on matches of that surface

Both are exposed as features; the model decides how to weigh them rather than
us fixing a blend ratio by hand.

K-factor is dynamic. New players move fast, established ones slowly:

    K = K0 / (matches + offset) ** decay

This is the standard tennis-Elo formulation (FiveThirtyEight-style). A flat K
either makes veterans too jumpy or newcomers too sluggish.
"""

from __future__ import annotations

import math
import sqlite3

BASE_RATING = 1500.0
K0 = 250.0
K_OFFSET = 5.0
K_DECAY = 0.4

# Best-of-five is a longer test and a better read on true strength, so a result
# there moves the rating slightly more than a best-of-three.
BO5_MULTIPLIER = 1.10


def expected(rating_a: float, rating_b: float) -> float:
    """Standard logistic expectation on the 400-point scale."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def k_factor(n_matches: int) -> float:
    return K0 / ((n_matches + K_OFFSET) ** K_DECAY)


def rebuild(conn: sqlite3.Connection, verbose: bool = True) -> dict:
    """
    Replay every match in date order, writing pre-match ratings onto each row.

    Run after any bulk load, and after each daily ingest. Idempotent -- it
    always starts from scratch, so ratings can't drift from repeated runs.
    """
    rows = conn.execute(
        """SELECT match_key, source, match_date, surface, best_of,
                  winner_id, loser_id, retirement
           FROM matches
           WHERE winner_id IS NOT NULL AND loser_id IS NOT NULL
           ORDER BY match_date, rowid"""
    ).fetchall()

    if verbose:
        print(f"  replaying {len(rows)} matches...")

    elo: dict[str, float] = {}
    elo_surf: dict[tuple[str, str], float] = {}
    played: dict[str, int] = {}
    played_surf: dict[tuple[str, str], int] = {}

    updates = []
    for i, r in enumerate(rows):
        if verbose and i % 5000 == 0 and i:
            print(f"    {i}/{len(rows)}")

        w, l = r["winner_id"], r["loser_id"]
        surf = r["surface"]

        rw = elo.get(w, BASE_RATING)
        rl = elo.get(l, BASE_RATING)

        if surf:
            sw = elo_surf.get((w, surf), BASE_RATING)
            sl = elo_surf.get((l, surf), BASE_RATING)
        else:
            sw = sl = None

        # record the PRE-match state, then update
        updates.append((rw, rl, sw, sl, r["match_key"], r["source"]))

        mult = BO5_MULTIPLIER if r["best_of"] == 5 else 1.0
        # A retirement is weak evidence about who was actually better, so it
        # moves ratings at half weight rather than counting as a full result.
        if r["retirement"]:
            mult *= 0.5

        exp_w = expected(rw, rl)
        kw = k_factor(played.get(w, 0)) * mult
        kl = k_factor(played.get(l, 0)) * mult
        elo[w] = rw + kw * (1.0 - exp_w)
        elo[l] = rl - kl * (1.0 - exp_w)
        played[w] = played.get(w, 0) + 1
        played[l] = played.get(l, 0) + 1

        if surf:
            exp_sw = expected(sw, sl)
            ksw = k_factor(played_surf.get((w, surf), 0)) * mult
            ksl = k_factor(played_surf.get((l, surf), 0)) * mult
            elo_surf[(w, surf)] = sw + ksw * (1.0 - exp_sw)
            elo_surf[(l, surf)] = sl - ksl * (1.0 - exp_sw)
            played_surf[(w, surf)] = played_surf.get((w, surf), 0) + 1
            played_surf[(l, surf)] = played_surf.get((l, surf), 0) + 1

    conn.executemany(
        """UPDATE matches
           SET w_elo = ?, l_elo = ?, w_elo_surface = ?, l_elo_surface = ?
           WHERE match_key = ? AND source = ?""",
        updates,
    )
    conn.commit()

    # current ratings, for serving predictions on matches not yet played
    conn.execute("""
        CREATE TABLE IF NOT EXISTS elo_current (
            player_id TEXT NOT NULL,
            surface   TEXT,                  -- NULL row = overall rating
            rating    REAL NOT NULL,
            matches   INTEGER NOT NULL,
            PRIMARY KEY (player_id, surface)
        )""")
    conn.execute("DELETE FROM elo_current")
    conn.executemany(
        "INSERT INTO elo_current VALUES (?, NULL, ?, ?)",
        [(p, rating, played.get(p, 0)) for p, rating in elo.items()],
    )
    conn.executemany(
        "INSERT INTO elo_current VALUES (?, ?, ?, ?)",
        [(p, s, rating, played_surf.get((p, s), 0))
         for (p, s), rating in elo_surf.items()],
    )
    conn.commit()

    top = sorted(elo.items(), key=lambda x: -x[1])[:5]
    return {
        "matches": len(rows),
        "players": len(elo),
        "top5": [(p, round(v)) for p, v in top],
    }


def current(conn: sqlite3.Connection, player_id: str,
            surface: str | None = None) -> tuple[float, int]:
    """Latest rating and match count. Falls back to base for unknown players."""
    row = conn.execute(
        "SELECT rating, matches FROM elo_current WHERE player_id = ? AND surface IS ?",
        (player_id, surface),
    ).fetchone()
    if row:
        return float(row["rating"]), int(row["matches"])
    return BASE_RATING, 0


def as_of(conn: sqlite3.Connection, player_id: str, date_iso: str,
          surface: str | None = None) -> tuple[float, int]:
    """
    Rating as it stood before `date_iso`, read from the last match row before
    that date. This is what backtests must use -- `current()` would leak.
    """
    if surface:
        sql = """SELECT CASE WHEN winner_id = :pid THEN w_elo_surface
                             ELSE l_elo_surface END AS r
                 FROM matches
                 WHERE (winner_id = :pid OR loser_id = :pid)
                   AND match_date < :d AND surface = :s
                   AND (CASE WHEN winner_id = :pid THEN w_elo_surface
                             ELSE l_elo_surface END) IS NOT NULL
                 ORDER BY match_date DESC, rowid DESC LIMIT 1"""
        params = {"pid": player_id, "d": date_iso, "s": surface}
    else:
        sql = """SELECT CASE WHEN winner_id = :pid THEN w_elo ELSE l_elo END AS r
                 FROM matches
                 WHERE (winner_id = :pid OR loser_id = :pid)
                   AND match_date < :d
                   AND (CASE WHEN winner_id = :pid THEN w_elo ELSE l_elo END) IS NOT NULL
                 ORDER BY match_date DESC, rowid DESC LIMIT 1"""
        params = {"pid": player_id, "d": date_iso}

    row = conn.execute(sql, params).fetchone()
    n = conn.execute(
        """SELECT COUNT(*) c FROM matches
           WHERE (winner_id = :pid OR loser_id = :pid) AND match_date < :d""",
        {"pid": player_id, "d": date_iso},
    ).fetchone()["c"]
    return (float(row["r"]) if row and row["r"] is not None else BASE_RATING), n


def win_probability(rating_a: float, rating_b: float) -> float:
    """Elo's own prediction. Used as a standalone baseline to beat."""
    return expected(rating_a, rating_b)
