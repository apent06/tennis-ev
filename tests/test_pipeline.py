"""
Smoke tests. Run: python test_pipeline.py

These cover the three things most likely to break silently:
  1. re-running ingest must not duplicate rows (idempotency)
  2. an ambiguous player name must NOT auto-resolve
  3. rank lookups must be point-in-time, not current
"""

import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import date

from ingest.adapters import FixtureSource
from ingest.crosswalk import confirm, pending_review, resolve
from ingest.daily import run
from ingest.db import (connect, init_db, make_match_key, normalize_name,
                       rank_as_of, upsert_ranking)
from ingest.freshness import head_to_head, player_form, quality_wins, surface_splits

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond):
    results.append((PASS if cond else FAIL, name))
    print(f"[{PASS if cond else FAIL}] {name}")


db_path = os.path.join(tempfile.mkdtemp(), "t.db")
conn = connect(db_path)
init_db(conn)

# -- name normalization -------------------------------------------------------
check("normalize folds accents/case/order",
      normalize_name("Raphaël COLLIGNON") == normalize_name("Collignon, Raphael"))
check("normalize drops bare initials",
      normalize_name("R. Collignon") == "collignon")

# -- match key stability ------------------------------------------------------
k1 = make_match_key("2026-08-11", "Cincinnati Masters", "Raphael Collignon", "Mariano Navone")
k2 = make_match_key("2026-08-11", "Cincinnati Masters", "Mariano Navone", "Raphael Collignon")
check("match_key is order-independent", k1 == k2)
k3 = make_match_key("2026-08-12", "Cincinnati Masters", "Raphael Collignon", "Mariano Navone")
check("match_key changes with date", k1 != k3)

# -- seed canonical players ---------------------------------------------------
players = [
    ("sack_001", "Raphael Collignon", "ATP"),
    ("sack_002", "Mariano Navone", "ATP"),
    ("sack_003", "Alexandre Muller", "ATP"),
    ("sack_004", "Benjamin Bonzi", "ATP"),
]
for pid, full, tour in players:
    conn.execute(
        "INSERT INTO players (player_id, full_name, norm_name, tour) VALUES (?,?,?,?)",
        (pid, full, normalize_name(full), tour))
conn.commit()

# -- crosswalk ----------------------------------------------------------------
rid = resolve(conn, "fixture", "p_collignon", "R. Collignon", "ATP")
check("exact-surname match resolves", rid == "sack_001")

# Ambiguity: two near-identical names must NOT auto-resolve.
conn.execute("INSERT INTO players VALUES ('sack_900','Juan Martin Lopez',?,'ATP',NULL,NULL)",
             (normalize_name("Juan Martin Lopez"),))
conn.execute("INSERT INTO players VALUES ('sack_901','Juan Manuel Lopez',?,'ATP',NULL,NULL)",
             (normalize_name("Juan Manuel Lopez"),))
conn.commit()
amb = resolve(conn, "fixture", "p_ambig", "J. Lopez", "ATP")
check("ambiguous name refuses to auto-resolve", amb is None)
check("ambiguous name lands in review queue",
      any(r["source_player_id"] == "p_ambig" for r in pending_review(conn)))
confirm(conn, "fixture", "p_ambig", "sack_900")
check("manual confirm takes effect",
      resolve(conn, "fixture", "p_ambig", "J. Lopez", "ATP") == "sack_900")

# -- point-in-time rankings ---------------------------------------------------
upsert_ranking(conn, "sack_001", "ATP", 55, 900, "2026-06-01")
upsert_ranking(conn, "sack_001", "ATP", 38, 1150, "2026-08-04")
check("rank_as_of returns the OLD rank for an old date",
      rank_as_of(conn, "sack_001", "2026-07-01") == 55)
check("rank_as_of returns the new rank after the update",
      rank_as_of(conn, "sack_001", "2026-08-11") == 38)
check("rank_as_of returns None before any snapshot",
      rank_as_of(conn, "sack_001", "2026-01-01") is None)

# -- ingest + idempotency -----------------------------------------------------
src = FixtureSource(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "fixtures", "sample.json"))
c1 = run(conn, src, date(2026, 8, 11), date(2026, 8, 12))
check(f"first run inserts 3 rows (got {c1['inserted']})", c1["inserted"] == 3)

c2 = run(conn, src, date(2026, 8, 11), date(2026, 8, 12))
total = conn.execute("SELECT COUNT(*) n FROM matches").fetchone()["n"]
check(f"re-run inserts nothing (got {c2['inserted']})", c2["inserted"] == 0)
check(f"row count still 3 after re-run (got {total})", total == 3)
check(f"re-run reports 0 meaningful updates (got {c2['updated']})", c2["updated"] == 0)

# amended score -> should UPDATE, not duplicate
conn.execute("UPDATE matches SET score='9-9 9-9' WHERE source_event_key='991001'")
conn.commit()
c3 = run(conn, src, date(2026, 8, 11), date(2026, 8, 11))
total_after = conn.execute("SELECT COUNT(*) n FROM matches").fetchone()["n"]
check(f"amended score updates in place (rows still 3, got {total_after})", total_after == 3)
check(f"amendment counted as an update (got {c3['updated']})", c3["updated"] >= 1)

# -- rank backfill onto matches ----------------------------------------------
row = conn.execute(
    "SELECT winner_rank FROM matches WHERE source_event_key='991001'").fetchone()
check(f"winner_rank backfilled point-in-time (got {row['winner_rank']})",
      row["winner_rank"] == 38)

# -- feature queries ----------------------------------------------------------
form = player_form(conn, "sack_001", n=10, as_of="2026-08-13")
check(f"form string built (got '{form['form_string']}')", form["form_string"] == "WW")
check("form carries staleness metadata", "days_since_last_match" in form)
check("form not flagged stale for recent matches", form["is_stale"] is False)

gap = player_form(conn, "sack_999_unknown", as_of="2026-08-13")
check("unknown player reports data_gap, not a guess", gap["data_gap"] is True)
check("unknown player flagged stale", gap["is_stale"] is True)

splits = surface_splits(conn, "sack_001", as_of="2026-08-13")
check(f"surface splits computed (got {splits})",
      splits.get("Hard", {}).get("wins") == 2)

h2h = head_to_head(conn, "sack_001", "sack_002", as_of="2026-08-13")
check(f"h2h found 1 meeting (got {len(h2h)})", len(h2h) == 1)

qw = quality_wins(conn, "sack_001", as_of="2026-08-13")
check("quality_wins buckets by opponent rank tier", isinstance(qw, dict) and "top10" in qw)

# -- backtest leakage guard ---------------------------------------------------
past = player_form(conn, "sack_001", as_of="2026-08-12")
check(f"as_of excludes same-day/future matches (got '{past['form_string']}')",
      past["form_string"] == "W")

print("\n" + "=" * 46)
n_fail = sum(1 for s, _ in results if s == FAIL)
print(f"{len(results) - n_fail}/{len(results)} passed")
if n_fail:
    for s, n in results:
        if s == FAIL:
            print(f"  FAILED: {n}")
    raise SystemExit(1)
