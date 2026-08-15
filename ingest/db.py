"""
Schema + upsert logic for the tennis match store.

Design notes:

1. `match_key` is a deterministic hash of (date, tour_level, sorted player names,
   tournament slug). It lets me dedup the same match arriving from different
   sources (Sackmann vs live API) without trusting either source's own ID.

2. `source_event_key` is the provider's own ID. Unique per (source, key). This
   is what makes re-pulling the same day idempotent.

3. `first_seen_at` / `last_updated_at` / `ingested_at` exist so I can measure
   ingestion lag per tour level instead of guessing at it. This is what turns
   "the data feels stale" into a number.

4. winner_rank / loser_rank are stored ON the match row. These are ranks as of
   match time, which makes opponent-quality features point-in-time correct with
   no join and no leakage.

5. rankings are versioned with valid_from/valid_to. Never UPDATE a rank in
   place, that silently corrupts every historical backtest.
"""

import hashlib
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_key           TEXT NOT NULL,
    source              TEXT NOT NULL,
    source_event_key    TEXT,

    match_date          TEXT NOT NULL,        -- ISO YYYY-mm-dd
    tour                TEXT,                 -- ATP / WTA
    tour_level          TEXT,                 -- G / M / A / C (challenger) / etc
    tournament          TEXT,
    round               TEXT,
    surface             TEXT,                 -- Hard / Clay / Grass / Carpet

    winner_id           TEXT,                 -- canonical player id
    loser_id            TEXT,
    winner_name         TEXT,
    loser_name          TEXT,
    winner_rank         INTEGER,              -- rank AS of this match
    loser_rank          INTEGER,
    score               TEXT,
    retirement          INTEGER DEFAULT 0,

    -- match shape
    best_of             INTEGER,              -- 3 or 5
    court               TEXT,                 -- Indoor / Outdoor
    w_games             INTEGER,              -- total games won by winner
    l_games             INTEGER,
    w_sets              INTEGER,
    l_sets              INTEGER,
    w_pts               INTEGER,              -- ranking points at match time
    l_pts               INTEGER,

    -- ratings as of before this match (filled by model/elo.py).
    -- Stored on the row so they are point-in-time correct with no join.
    w_elo               REAL,
    l_elo               REAL,
    w_elo_surface       REAL,
    l_elo_surface       REAL,

    -- serve/return detail. Empty for tennis-data.co.uk, which does not carry
    -- it. Present so a stats-bearing source can populate them without a
    -- migration; every consumer treats NULL as 'unavailable', never as zero.
    w_ace               INTEGER,
    l_ace               INTEGER,
    w_df                INTEGER,
    l_df                INTEGER,
    w_svpt              INTEGER,              -- service points played
    l_svpt              INTEGER,
    w_1st_in            INTEGER,
    l_1st_in            INTEGER,
    w_1st_won           INTEGER,
    l_1st_won           INTEGER,
    w_2nd_won           INTEGER,
    l_2nd_won           INTEGER,
    w_bp_saved          INTEGER,
    l_bp_saved          INTEGER,
    w_bp_faced          INTEGER,
    l_bp_faced          INTEGER,

    first_seen_at       TEXT NOT NULL,
    last_updated_at     TEXT NOT NULL,
    ingested_at         TEXT NOT NULL,

    PRIMARY KEY (match_key, source)
);

CREATE INDEX IF NOT EXISTS idx_matches_date     ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_matches_winner   ON matches(winner_id, match_date);
CREATE INDEX IF NOT EXISTS idx_matches_loser    ON matches(loser_id, match_date);
CREATE INDEX IF NOT EXISTS idx_matches_surface  ON matches(surface);
CREATE UNIQUE INDEX IF NOT EXISTS idx_matches_src_event
    ON matches(source, source_event_key) WHERE source_event_key IS NOT NULL;

-- Versioned ranking snapshots. valid_to is NULL == currently in force.
CREATE TABLE IF NOT EXISTS rankings (
    player_id       TEXT NOT NULL,
    tour            TEXT NOT NULL,
    rank            INTEGER NOT NULL,
    points          INTEGER,
    valid_from      TEXT NOT NULL,
    valid_to        TEXT,
    ingested_at     TEXT NOT NULL,
    PRIMARY KEY (player_id, valid_from)
);

CREATE INDEX IF NOT EXISTS idx_rankings_lookup ON rankings(player_id, valid_from, valid_to);

CREATE TABLE IF NOT EXISTS players (
    player_id   TEXT PRIMARY KEY,
    full_name   TEXT NOT NULL,
    norm_name   TEXT NOT NULL,
    tour        TEXT,
    dob         TEXT,                 -- age is computed at match time, never stored
    country     TEXT
);

CREATE INDEX IF NOT EXISTS idx_players_norm ON players(norm_name);

-- Maps a provider's player id onto the canonical player_id.
-- `reviewed` = a human confirmed it. Unreviewed low-confidence rows are the
-- single most likely source of a silently-wrong prediction.
CREATE TABLE IF NOT EXISTS player_crosswalk (
    source              TEXT NOT NULL,
    source_player_id    TEXT NOT NULL,
    source_name         TEXT,
    player_id           TEXT,
    confidence          REAL,
    reviewed            INTEGER DEFAULT 0,
    created_at          TEXT,
    PRIMARY KEY (source, source_player_id)
);

-- One row per ingest run, so freshness is auditable after the fact.
CREATE TABLE IF NOT EXISTS ingest_log (
    run_id          TEXT PRIMARY KEY,
    source          TEXT,
    window_start    TEXT,
    window_end      TEXT,
    rows_seen       INTEGER,
    rows_inserted   INTEGER,
    rows_updated    INTEGER,
    started_at      TEXT,
    finished_at     TEXT,
    status          TEXT,
    error           TEXT
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# Columns added after the first release. migrate() adds any that are missing
# so an existing database can be upgraded in place without a reload.
LATER_COLUMNS = [
    ("best_of", "INTEGER"), ("court", "TEXT"),
    ("w_games", "INTEGER"), ("l_games", "INTEGER"),
    ("w_sets", "INTEGER"), ("l_sets", "INTEGER"),
    ("w_pts", "INTEGER"), ("l_pts", "INTEGER"),
    ("w_elo", "REAL"), ("l_elo", "REAL"),
    ("w_elo_surface", "REAL"), ("l_elo_surface", "REAL"),
    ("w_ace", "INTEGER"), ("l_ace", "INTEGER"),
    ("w_df", "INTEGER"), ("l_df", "INTEGER"),
    ("w_svpt", "INTEGER"), ("l_svpt", "INTEGER"),
    ("w_1st_in", "INTEGER"), ("l_1st_in", "INTEGER"),
    ("w_1st_won", "INTEGER"), ("l_1st_won", "INTEGER"),
    ("w_2nd_won", "INTEGER"), ("l_2nd_won", "INTEGER"),
    ("w_bp_saved", "INTEGER"), ("l_bp_saved", "INTEGER"),
    ("w_bp_faced", "INTEGER"), ("l_bp_faced", "INTEGER"),
]


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Add any columns missing from an older database. Safe to re-run."""
    have = {r[1] for r in conn.execute("PRAGMA table_info(matches)")}
    added = []
    for name, coltype in LATER_COLUMNS:
        if name not in have:
            conn.execute(f"ALTER TABLE matches ADD COLUMN {name} {coltype}")
            added.append(name)
    conn.commit()
    return added


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    migrate(conn)
    conn.commit()


def normalize_name(name: str) -> str:
    """
    Fold a player name to a comparable form.

    Providers disagree wildly: 'R. Collignon', 'Raphael Collignon',
    'Collignon R.', 'Raphaël Collignon'. We strip accents, punctuation and
    case, then sort tokens so word order stops mattering.
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    tokens = [t for t in s.split() if t]
    # Drop bare initials, they carry no matching signal and hurt token overlap.
    tokens = [t for t in tokens if len(t) > 1]
    return " ".join(sorted(tokens))


def slugify(text: str) -> str:
    if not text:
        return ""
    s = unicodedata.normalize("NFKD", text)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def make_match_key(match_date: str, tournament: str, name_a: str, name_b: str) -> str:
    """
    Cross-source stable identity for a match.

    Deliberately does not include round or score: those are the fields most
    likely to be reported inconsistently (or amended after the fact), and a key
    that changes on amendment defeats the whole point of an upsert.
    """
    players = sorted([normalize_name(name_a), normalize_name(name_b)])
    raw = "|".join([match_date, slugify(tournament), players[0], players[1]])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


UPSERT_SQL = """
INSERT INTO matches (
    match_key, source, source_event_key,
    match_date, tour, tour_level, tournament, round, surface,
    winner_id, loser_id, winner_name, loser_name,
    winner_rank, loser_rank, score, retirement,
    best_of, court, w_games, l_games, w_sets, l_sets, w_pts, l_pts,
    first_seen_at, last_updated_at, ingested_at
) VALUES (
    :match_key, :source, :source_event_key,
    :match_date, :tour, :tour_level, :tournament, :round, :surface,
    :winner_id, :loser_id, :winner_name, :loser_name,
    :winner_rank, :loser_rank, :score, :retirement,
    :best_of, :court, :w_games, :l_games, :w_sets, :l_sets, :w_pts, :l_pts,
    :ts, :ts, :ts
)
ON CONFLICT(match_key, source) DO UPDATE SET
    source_event_key = excluded.source_event_key,
    round            = excluded.round,
    surface          = COALESCE(excluded.surface, matches.surface),
    winner_id        = COALESCE(excluded.winner_id, matches.winner_id),
    loser_id         = COALESCE(excluded.loser_id, matches.loser_id),
    winner_name      = excluded.winner_name,
    loser_name       = excluded.loser_name,
    winner_rank      = COALESCE(excluded.winner_rank, matches.winner_rank),
    loser_rank       = COALESCE(excluded.loser_rank, matches.loser_rank),
    score            = excluded.score,
    retirement       = excluded.retirement,
    best_of          = COALESCE(excluded.best_of, matches.best_of),
    court            = COALESCE(excluded.court, matches.court),
    w_games          = COALESCE(excluded.w_games, matches.w_games),
    l_games          = COALESCE(excluded.l_games, matches.l_games),
    w_sets           = COALESCE(excluded.w_sets, matches.w_sets),
    l_sets           = COALESCE(excluded.l_sets, matches.l_sets),
    w_pts            = COALESCE(excluded.w_pts, matches.w_pts),
    l_pts            = COALESCE(excluded.l_pts, matches.l_pts),
    ingested_at      = excluded.ingested_at,
    -- only bump last_updated_at when something meaningful changed, so this
    -- column stays a real change-detector rather than a re-run counter
    last_updated_at  = CASE
        WHEN matches.score IS NOT excluded.score
          OR matches.winner_name IS NOT excluded.winner_name
          OR matches.retirement IS NOT excluded.retirement
        THEN excluded.ingested_at
        ELSE matches.last_updated_at
    END
WHERE TRUE;
"""


def upsert_matches(conn: sqlite3.Connection, rows: list[dict], source: str) -> dict:
    """Idempotent bulk upsert. Returns counts for the ingest log."""
    ts = now_iso()
    inserted = updated = 0

    for r in rows:
        key = make_match_key(
            r["match_date"], r.get("tournament", ""), r["winner_name"], r["loser_name"]
        )
        existing = conn.execute(
            "SELECT last_updated_at FROM matches WHERE match_key=? AND source=?",
            (key, source),
        ).fetchone()

        payload = {
            "match_key": key,
            "source": source,
            "source_event_key": r.get("source_event_key"),
            "match_date": r["match_date"],
            "tour": r.get("tour"),
            "tour_level": r.get("tour_level"),
            "tournament": r.get("tournament"),
            "round": r.get("round"),
            "surface": r.get("surface"),
            "winner_id": r.get("winner_id"),
            "loser_id": r.get("loser_id"),
            "winner_name": r["winner_name"],
            "loser_name": r["loser_name"],
            "winner_rank": r.get("winner_rank"),
            "loser_rank": r.get("loser_rank"),
            "score": r.get("score"),
            "retirement": int(bool(r.get("retirement", 0))),
            "best_of": r.get("best_of"), "court": r.get("court"),
            "w_games": r.get("w_games"), "l_games": r.get("l_games"),
            "w_sets": r.get("w_sets"), "l_sets": r.get("l_sets"),
            "w_pts": r.get("w_pts"), "l_pts": r.get("l_pts"),
            "ts": ts,
        }
        conn.execute(UPSERT_SQL, payload)

        if existing is None:
            inserted += 1
        else:
            after = conn.execute(
                "SELECT last_updated_at FROM matches WHERE match_key=? AND source=?",
                (key, source),
            ).fetchone()
            if after["last_updated_at"] != existing["last_updated_at"]:
                updated += 1

    conn.commit()
    return {"seen": len(rows), "inserted": inserted, "updated": updated}


def upsert_ranking(
    conn: sqlite3.Connection, player_id: str, tour: str, rank: int,
    points: int | None, valid_from: str,
) -> None:
    """
    Close out the previous open snapshot, then open a new one. This is what
    keeps historical backtests honest, I can always ask 'what was this
    player's rank on date X' and get the answer as it was known THEN.
    """
    conn.execute(
        """UPDATE rankings SET valid_to = ?
           WHERE player_id = ? AND valid_to IS NULL AND valid_from < ?""",
        (valid_from, player_id, valid_from),
    )
    conn.execute(
        """INSERT INTO rankings (player_id, tour, rank, points, valid_from, valid_to, ingested_at)
           VALUES (?, ?, ?, ?, ?, NULL, ?)
           ON CONFLICT(player_id, valid_from) DO UPDATE SET
               rank = excluded.rank, points = excluded.points""",
        (player_id, tour, rank, points, valid_from, now_iso()),
    )


def rank_as_of(conn: sqlite3.Connection, player_id: str, as_of: str) -> int | None:
    """Point-in-time rank lookup. Use this, never 'SELECT current rank'."""
    row = conn.execute(
        """SELECT rank FROM rankings
           WHERE player_id = ? AND valid_from <= ?
             AND (valid_to IS NULL OR valid_to > ?)
           ORDER BY valid_from DESC LIMIT 1""",
        (player_id, as_of, as_of),
    ).fetchone()
    return row["rank"] if row else None
