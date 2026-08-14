"""
Project CLI.

    python cli.py init                          # create schema
    python cli.py synth                         # generate fake data (no network)
    python cli.py load-tennisdata data/td/      # backfill from tennis-data.co.uk
    python cli.py load-sackmann data/sackmann/  # backfill from a local clone
    python cli.py ingest --days 3               # daily live pull (needs API key)
    python cli.py train                         # train + calibrate
    python cli.py backtest                      # walk-forward vs closing odds
    python cli.py check                         # freshness report
    python cli.py review                        # unresolved player names
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, timedelta

from ingest.crosswalk import pending_review
from ingest.db import connect, init_db
from ingest.freshness import check_slo, observed_lag

DB = os.environ.get("TENNIS_DB", "tennis.db")


def cmd_init(args):
    conn = connect(args.db)
    init_db(conn)
    print(f"schema ready: {args.db}")


def cmd_synth(args):
    from ingest.loaders import generate_synthetic
    conn = connect(args.db)
    init_db(conn)
    c = generate_synthetic(conn, n_players=args.players, n_matches=args.matches)
    print(f"synthetic: {c['inserted']} matches, {args.players} players")


def cmd_load_tennisdata(args):
    from ingest.loaders import load_tennis_data_dir
    conn = connect(args.db)
    init_db(conn)
    print(load_tennis_data_dir(conn, args.folder))


def cmd_load_sackmann(args):
    from ingest.loaders import load_sackmann_dir
    conn = connect(args.db)
    init_db(conn)
    print(load_sackmann_dir(conn, args.folder, args.tour))


def cmd_ingest(args):
    from datetime import date as _d
    from ingest.adapters import ApiTennisSource, FixtureSource
    from ingest.daily import run
    conn = connect(args.db)
    init_db(conn)
    src = FixtureSource(args.fixture) if args.fixture else ApiTennisSource()
    end = _d.today()
    start = end - timedelta(days=args.days)
    print(run(conn, src, start, end))


def cmd_elo(args):
    from ingest.db import migrate
    from model.elo import rebuild
    conn = connect(args.db)
    migrate(conn)
    r = rebuild(conn)
    print(f"elo rebuilt: {r['matches']} matches, {r['players']} players")
    for pid, rating in r["top5"]:
        name = conn.execute("SELECT full_name FROM players WHERE player_id=?",
                            (pid,)).fetchone()
        print(f"   {rating:>5}  {name['full_name'] if name else pid}")


def cmd_baselines(args):
    from model.baselines import main as run
    run()


def cmd_migrate(args):
    from ingest.db import migrate
    conn = connect(args.db)
    added = migrate(conn)
    print("added columns:", added or "none (already up to date)")


def cmd_train(args):
    from model.train import train
    conn = connect(args.db)
    end = args.end or date.today().isoformat()
    start = args.start or (date.today() - timedelta(days=args.window)).isoformat()
    train(conn, start, end, out_path=args.out)


def cmd_backtest(args):
    from model.backtest import walk_forward
    conn = connect(args.db)
    end = args.end or date.today().isoformat()
    start = args.start or (date.today() - timedelta(days=args.window)).isoformat()
    res = walk_forward(conn, start, end, edge_threshold=args.edge)
    print(json.dumps(res, indent=2))


def cmd_check(args):
    conn = connect(args.db)
    for r in observed_lag(conn):
        print(f"  {r['tour_level']:>3}  n={r['n']:<6} median={r['median_lag_days']}d "
              f"p90={r['p90_lag_days']}d")
    b = check_slo(conn)
    print("SLO: OK" if not b else f"SLO BREACH: {b}")


def cmd_review(args):
    conn = connect(args.db)
    rows = pending_review(conn)
    if not rows:
        print("nothing pending")
    for r in rows:
        print(f"  [{r['confidence']:.2f}] {r['source']}:{r['source_player_id']} -> {r['source_name']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init").set_defaults(fn=cmd_init)

    s = sub.add_parser("synth")
    s.add_argument("--players", type=int, default=120)
    s.add_argument("--matches", type=int, default=4000)
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("load-tennisdata")
    s.add_argument("folder")
    s.set_defaults(fn=cmd_load_tennisdata)

    s = sub.add_parser("load-sackmann")
    s.add_argument("folder")
    s.add_argument("--tour", default="ATP")
    s.set_defaults(fn=cmd_load_sackmann)

    s = sub.add_parser("ingest")
    s.add_argument("--days", type=int, default=3)
    s.add_argument("--fixture")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("train")
    s.add_argument("--start")
    s.add_argument("--end")
    s.add_argument("--window", type=int, default=1095)
    s.add_argument("--out", default="model.pkl")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("backtest")
    s.add_argument("--start")
    s.add_argument("--end")
    s.add_argument("--window", type=int, default=1095)
    s.add_argument("--edge", type=float, default=0.04)
    s.set_defaults(fn=cmd_backtest)

    sub.add_parser("elo").set_defaults(fn=cmd_elo)
    sub.add_parser("baselines").set_defaults(fn=cmd_baselines)
    sub.add_parser("migrate").set_defaults(fn=cmd_migrate)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    sub.add_parser("review").set_defaults(fn=cmd_review)

    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
