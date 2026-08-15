"""
Daily ingest entry point.

    python -m ingest.daily --db tennis.db --days 3
    python -m ingest.daily --db tennis.db --backfill 365
    python -m ingest.daily --db tennis.db --check
    python -m ingest.daily --db tennis.db --review

Why a rolling window rather than just yesterday: results get amended after
first publication (late finishes, retirements, corrections, walkovers). Re-pulling
the last few days with idempotent upserts catches those for free.
"""

from __future__ import annotations

import argparse
import uuid
from datetime import date, timedelta

from .adapters import ApiTennisSource, FixtureSource
from .crosswalk import pending_review, resolve
from .db import connect, init_db, now_iso, rank_as_of, upsert_matches
from .freshness import check_slo, observed_lag


def attach_ids_and_ranks(conn, rows: list[dict], source: str) -> list[dict]:
    """Resolve player identities, then backfill rank-as-of-match-date."""
    for r in rows:
        r["winner_id"] = resolve(conn, source, r.get("source_winner_id"),
                                 r["winner_name"], r.get("tour"))
        r["loser_id"] = resolve(conn, source, r.get("source_loser_id"),
                                r["loser_name"], r.get("tour"))
        if r.get("winner_rank") is None and r["winner_id"]:
            r["winner_rank"] = rank_as_of(conn, r["winner_id"], r["match_date"])
        if r.get("loser_rank") is None and r["loser_id"]:
            r["loser_rank"] = rank_as_of(conn, r["loser_id"], r["match_date"])
    return rows


def run(conn, source, start: date, end: date) -> dict:
    run_id = str(uuid.uuid4())[:8]
    started = now_iso()
    conn.execute(
        """INSERT INTO ingest_log (run_id, source, window_start, window_end, started_at, status)
           VALUES (?,?,?,?,?,'running')""",
        (run_id, source.name, start.isoformat(), end.isoformat(), started),
    )
    conn.commit()

    try:
        rows = source.fetch_range(start, end)
        rows = attach_ids_and_ranks(conn, rows, source.name)
        counts = upsert_matches(conn, rows, source.name)
        conn.execute(
            """UPDATE ingest_log SET rows_seen=?, rows_inserted=?, rows_updated=?,
                   finished_at=?, status='ok' WHERE run_id=?""",
            (counts["seen"], counts["inserted"], counts["updated"], now_iso(), run_id),
        )
        conn.commit()
        return counts
    except Exception as exc:
        conn.execute(
            "UPDATE ingest_log SET finished_at=?, status='error', error=? WHERE run_id=?",
            (now_iso(), str(exc)[:500], run_id),
        )
        conn.commit()
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="tennis.db")
    ap.add_argument("--days", type=int, default=3, help="rolling re-pull window")
    ap.add_argument("--backfill", type=int, help="one-off backfill, N days back")
    ap.add_argument("--fixture", help="use a JSON fixture instead of the live API")
    ap.add_argument("--check", action="store_true", help="freshness report only")
    ap.add_argument("--review", action="store_true", help="list unresolved players")
    args = ap.parse_args()

    conn = connect(args.db)
    init_db(conn)

    if args.check:
        for row in observed_lag(conn):
            print(f"  {row['tour_level']:>3}  n={row['n']:<6} "
                  f"median={row['median_lag_days']}d  p90={row['p90_lag_days']}d")
        breaches = check_slo(conn)
        print("\nSLO: OK" if not breaches else f"\nSLO BREACH: {breaches}")
        return

    if args.review:
        rows = pending_review(conn)
        if not rows:
            print("No players pending review.")
        for r in rows:
            print(f"  [{r['confidence']:.2f}] {r['source']}:{r['source_player_id']} "
                  f"-> {r['source_name']}")
        return

    source = FixtureSource(args.fixture) if args.fixture else ApiTennisSource()
    end = date.today()
    start = end - timedelta(days=args.backfill if args.backfill else args.days)

    counts = run(conn, source, start, end)
    print(f"{source.name}: seen={counts['seen']} "
          f"inserted={counts['inserted']} updated={counts['updated']}")

    breaches = check_slo(conn)
    if breaches:
        print(f"WARNING freshness SLO breach: {breaches}")


if __name__ == "__main__":
    main()
