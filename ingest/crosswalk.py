"""
Player identity resolution between a provider's IDs and our canonical ids.

This is the highest-risk component in the pipeline. A wrong mapping doesn't
throw -- it silently serves a confident prediction built on the wrong player's
form. So the design is deliberately conservative:

  - exact normalized-name match          -> auto-accept (confidence 1.0)
  - strong fuzzy match, unambiguous      -> auto-accept (confidence >= 0.90)
  - anything weaker, or ambiguous        -> park it for manual review, and
                                            REFUSE to resolve until reviewed

Unresolved players surface as a data gap, not as a guess.
"""

from __future__ import annotations

import sqlite3
from difflib import SequenceMatcher

from .db import normalize_name, now_iso

AUTO_ACCEPT = 0.90
AMBIGUITY_MARGIN = 0.05   # top two candidates this close => treat as ambiguous


SUBSET_SCORE = 0.92   # one name is an abbreviation of the other


def _similarity(a: str, b: str) -> float:
    """
    Similarity between two normalized names.

    The dominant real-world case is abbreviation: feeds emit 'R. Collignon'
    while our canonical record says 'Raphael Collignon'. After normalization
    those become 'collignon' vs 'collignon raphael' -- a SUBSET, not a near
    string match. Jaccard punishes that (0.5) and edit distance punishes it
    (0.69), so both understate a match that is actually very likely.

    So containment is the primary signal, not overlap. The false positives it
    invites -- a bare surname matching several players -- are caught by the
    ambiguity margin in resolve(), which is the correct place for that check.
    A common surname SHOULD go to manual review rather than be scored away.
    """
    if not a or not b:
        return 0.0
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    if ta == tb:
        return 1.0

    shared = ta & tb
    containment = len(shared) / min(len(ta), len(tb))
    if containment == 1.0:
        # Every token of the shorter name appears in the longer one.
        return SUBSET_SCORE

    jaccard = len(shared) / len(ta | tb)
    seq = SequenceMatcher(None, a, b).ratio()
    return 0.5 * containment + 0.3 * jaccard + 0.2 * seq


def candidates(conn: sqlite3.Connection, name: str, tour: str | None = None,
               limit: int = 5) -> list[tuple[str, str, float]]:
    norm = normalize_name(name)
    sql = "SELECT player_id, full_name, norm_name FROM players"
    params: tuple = ()
    if tour:
        sql += " WHERE tour = ?"
        params = (tour,)
    scored = [
        (r["player_id"], r["full_name"], _similarity(norm, r["norm_name"]))
        for r in conn.execute(sql, params)
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:limit]


def resolve(conn: sqlite3.Connection, source: str, source_player_id: str | None,
            name: str, tour: str | None = None) -> str | None:
    """
    Return a canonical player_id, or None if unresolved.

    None is a legitimate, expected outcome. Downstream must treat it as a data
    gap and degrade confidence -- never as a reason to fall back to name
    matching at query time.
    """
    key_id = source_player_id or f"name:{normalize_name(name)}"

    cached = conn.execute(
        "SELECT player_id, reviewed, confidence FROM player_crosswalk WHERE source=? AND source_player_id=?",
        (source, key_id),
    ).fetchone()
    if cached is not None:
        if cached["player_id"] and (cached["reviewed"] or cached["confidence"] >= AUTO_ACCEPT):
            return cached["player_id"]
        return None  # parked for review

    cands = candidates(conn, name, tour)
    best_id, best_conf = None, 0.0
    if cands:
        top_id, _, top_score = cands[0]
        runner_up = cands[1][2] if len(cands) > 1 else 0.0
        ambiguous = (top_score - runner_up) < AMBIGUITY_MARGIN and len(cands) > 1
        if top_score >= AUTO_ACCEPT and not ambiguous:
            best_id, best_conf = top_id, top_score
        else:
            best_id, best_conf = None, top_score

    conn.execute(
        """INSERT INTO player_crosswalk
               (source, source_player_id, source_name, player_id, confidence, reviewed, created_at)
           VALUES (?, ?, ?, ?, ?, 0, ?)
           ON CONFLICT(source, source_player_id) DO UPDATE SET
               confidence = excluded.confidence,
               player_id  = COALESCE(player_crosswalk.player_id, excluded.player_id)""",
        (source, key_id, name, best_id, best_conf, now_iso()),
    )
    conn.commit()
    return best_id


def pending_review(conn: sqlite3.Connection, limit: int = 50) -> list[sqlite3.Row]:
    """Work queue for you. Run this after every backfill."""
    return conn.execute(
        """SELECT source, source_player_id, source_name, confidence
           FROM player_crosswalk
           WHERE reviewed = 0 AND (player_id IS NULL OR confidence < ?)
           ORDER BY confidence DESC LIMIT ?""",
        (AUTO_ACCEPT, limit),
    ).fetchall()


def confirm(conn: sqlite3.Connection, source: str, source_player_id: str,
            player_id: str) -> None:
    conn.execute(
        """UPDATE player_crosswalk
           SET player_id = ?, reviewed = 1, confidence = 1.0
           WHERE source = ? AND source_player_id = ?""",
        (player_id, source, source_player_id),
    )
    conn.commit()
