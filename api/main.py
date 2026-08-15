"""
FastAPI serving layer.

/predict never returns a bare probability.
Every response carries the freshness of the inputs it was built from, and a
confidence label that degrades when data is stale or missing. If either player
has no usable history, it refuses outright rather than returning a number built
on default values.

    uvicorn api.main:app --reload
    open http://localhost:8000/docs
"""

from __future__ import annotations

import os
import sqlite3
from datetime import date

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ingest.db import connect, normalize_name
from ingest.freshness import check_slo, observed_lag, player_form
from model.features import FEATURE_NAMES, build_features
from model.train import load_model

DB_PATH = os.environ.get("TENNIS_DB", "tennis.db")
MODEL_PATH = os.environ.get("TENNIS_MODEL", "model.pkl")

app = FastAPI(title="Tennis EV API", version="1.0.0")
_model_cache: dict | None = None


def db() -> sqlite3.Connection:
    return connect(DB_PATH)


def model():
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(503, f"No model at {MODEL_PATH}. Run training first.")
        _model_cache = load_model(MODEL_PATH)
    return _model_cache


class Prediction(BaseModel):
    p1: str
    p2: str
    surface: str | None
    as_of: str
    p1_win_probability: float
    fair_odds_p1: float
    fair_odds_p2: float
    confidence: str
    warnings: list[str]
    freshness: dict
    explanation: dict | None = None
    comparison: list[dict] | None = None


def resolve_player(conn, name_or_id: str) -> tuple[str, str]:
    row = conn.execute(
        "SELECT player_id, full_name FROM players WHERE player_id = ?", (name_or_id,)
    ).fetchone()
    if row:
        return row["player_id"], row["full_name"]

    norm = normalize_name(name_or_id)
    rows = conn.execute(
        "SELECT player_id, full_name, tour FROM players WHERE norm_name = ?", (norm,)
    ).fetchall()
    if len(rows) == 1:
        return rows[0]["player_id"], rows[0]["full_name"]
    if len(rows) > 1:
        # Same display name, different players, most often one per tour, since
        # the source stores names as surname-plus-initial with no first name.
        today = date.today().isoformat()
        cands = []
        for r in rows:
            n = conn.execute(
                """SELECT COUNT(*) c FROM matches
                   WHERE winner_id = :pid OR loser_id = :pid""",
                {"pid": r["player_id"]},
            ).fetchone()["c"]
            cands.append({
                "id": r["player_id"],
                "full_name": r["full_name"],
                "tour": r["tour"],
                "rank": latest_known_rank(conn, r["player_id"], today),
                "matches": n,
            })
        cands.sort(key=lambda x: -x["matches"])
        raise HTTPException(409, {
            "error": f"more than one player is called {name_or_id}",
            "detail": "Pick one from the dropdown, which sends the exact player.",
            "candidates": cands,
        })

    like = conn.execute(
        "SELECT player_id, full_name FROM players WHERE norm_name LIKE ? LIMIT 10",
        (f"%{norm}%",),
    ).fetchall()
    raise HTTPException(404, {
        "error": f"player not found: {name_or_id}",
        "did_you_mean": [dict(r) for r in like],
    })


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the browser interface."""
    path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if not os.path.exists(path):
        raise HTTPException(404, "index.html not found")
    with open(path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/players")
def players(q: str = Query("", description="name fragment"), limit: int = 12):
    """
    Name search for the autocomplete box.

    Returns rank and last-match context alongside the name. The source data
    stores names as surname-plus-initial only ('Collignon R.'), so there are no
    first names to show, this context is what lets you tell two similar
    entries apart.
    """
    conn = db()
    if len(q.strip()) < 2:
        return {"players": []}
    norm = normalize_name(q)
    rows = conn.execute(
        """SELECT player_id, full_name, tour FROM players
           WHERE norm_name LIKE ? OR lower(full_name) LIKE ?
           ORDER BY length(full_name) LIMIT ?""",
        (f"%{norm}%", f"%{q.strip().lower()}%", limit),
    ).fetchall()

    today = date.today().isoformat()
    out = []
    for r in rows:
        last = conn.execute(
            """SELECT match_date, tournament FROM matches
               WHERE winner_id = :pid OR loser_id = :pid
               ORDER BY match_date DESC LIMIT 1""",
            {"pid": r["player_id"]},
        ).fetchone()
        n_matches = conn.execute(
            """SELECT COUNT(*) c FROM matches
               WHERE winner_id = :pid OR loser_id = :pid""",
            {"pid": r["player_id"]},
        ).fetchone()["c"]
        out.append({
            "id": r["player_id"],
            "name": r["full_name"],
            "tour": r["tour"],
            "rank": latest_known_rank(conn, r["player_id"], today),
            "last_match": last["match_date"] if last else None,
            "last_event": last["tournament"] if last else None,
            "matches": n_matches,
        })
    # most active first, the player you meant is usually the busier one
    out.sort(key=lambda x: -x["matches"])
    return {"players": out}


@app.get("/health")
def health():
    conn = db()
    row = conn.execute(
        "SELECT COUNT(*) n, MAX(match_date) newest FROM matches").fetchone()
    breaches = check_slo(conn)
    return {
        "status": "degraded" if breaches else "ok",
        "matches": row["n"],
        "newest_match": row["newest"],
        "slo_breaches": breaches,
        "model_loaded": os.path.exists(MODEL_PATH),
    }


@app.get("/freshness")
def freshness():
    conn = db()
    return {"lag_by_tour_level": observed_lag(conn), "slo_breaches": check_slo(conn)}


@app.get("/player/{name}")
def player(name: str, as_of: str | None = None):
    conn = db()
    pid, full = resolve_player(conn, name)
    as_of = as_of or date.today().isoformat()
    form = player_form(conn, pid, n=10, as_of=as_of)
    return {"player_id": pid, "name": full, **form}


def latest_known_rank(conn, player_id: str, as_of: str) -> int | None:
    """
    Most recent rank available for a player.

    Prefers the versioned rankings table, but falls back to the rank recorded
    on their latest match. Tennis-Data stores rank per match rather than as
    weekly snapshots, so for that source the rankings table is empty and this
    fallback is the only thing that works.
    """
    from model.features import rank_as_of
    r = rank_as_of(conn, player_id, as_of)
    if r:
        return r
    row = conn.execute(
        """SELECT CASE WHEN winner_id = :pid THEN winner_rank ELSE loser_rank END AS rk
           FROM matches
           WHERE (winner_id = :pid OR loser_id = :pid) AND match_date < :as_of
             AND CASE WHEN winner_id = :pid THEN winner_rank ELSE loser_rank END IS NOT NULL
           ORDER BY match_date DESC LIMIT 1""",
        {"pid": player_id, "as_of": as_of},
    ).fetchone()
    return row["rk"] if row else None


def _predict_core(p1: str, p2: str, surface: str | None = None,
                  tour_level: str | None = None, as_of: str | None = None,
                  best_of: int | None = None, court: str | None = None) -> Prediction:
    """
    Plain-Python prediction logic.

    Kept separate from the route handler because FastAPI's Query(...) defaults
    are Query objects, not values, when a handler is called directly from other
    Python code. Routes stay thin; every internal caller uses this.
    """
    conn = db()
    as_of = as_of or date.today().isoformat()
    id1, name1 = resolve_player(conn, p1)
    id2, name2 = resolve_player(conn, p2)
    if id1 == id2:
        raise HTTPException(400, "p1 and p2 are the same player")

    fb = build_features(conn, id1, id2, surface, as_of, tour_level,
                        best_of, court)
    meta = fb["meta"]

    # Refuse rather than guess. A confident number built on default features is
    # worse than no number.
    if meta["p1_gap"] or meta["p2_gap"]:
        missing = [n for n, g in ((name1, meta["p1_gap"]), (name2, meta["p2_gap"])) if g]
        raise HTTPException(422, {
            "error": "insufficient match history",
            "players_without_data": missing,
            "detail": "Refusing to predict from default features.",
        })

    m = model()
    x = [[fb["features"][n] for n in FEATURE_NAMES]]
    from model.train import predict_calibrated
    p = float(predict_calibrated(m["model"], x)[0])

    warnings, confidence = [], "high"

    for who, tier, days, stretched, frm, to in (
        (name1, meta["p1_stale_tier"], meta["p1_days_since"],
         meta["p1_window_stretched"], meta["p1_form_from"], meta["p1_form_to"]),
        (name2, meta["p2_stale_tier"], meta["p2_days_since"],
         meta["p2_window_stretched"], meta["p2_form_from"], meta["p2_form_to"]),
    ):
        # A long gap is not the same as a slightly old record. Say which.
        if tier == "hard":
            warnings.append(
                f"{who} hasn't played in {days} days — form from before a long "
                f"break may not carry over")
            confidence = "low"
        elif tier == "soft":
            warnings.append(f"{who}'s last match was {days} days ago")
            if confidence == "high":
                confidence = "medium"
        if stretched and frm and to:
            warnings.append(
                f"{who}'s last 10 matches span {frm} to {to} — a thin schedule, "
                f"so this is less 'recent form' than it looks")
            confidence = "low"

    if fb["features"]["min_season_matches"] < 10:
        warnings.append("thin season history for at least one player")
        confidence = "low"
    if fb["features"]["min_surface_n"] < 5:
        warnings.append(f"limited {surface or 'surface'} sample")
        if confidence == "high":
            confidence = "medium"
    r1 = meta["p1_rank"] or latest_known_rank(conn, id1, as_of)
    r2 = meta["p2_rank"] or latest_known_rank(conn, id2, as_of)

    if not r1 or not r2:
        warnings.append("missing ranking for at least one player")

    from model.explain import explain, side_by_side
    try:
        expl = explain(m["model"], fb["features"])
        comp = side_by_side(conn, id1, id2, surface, as_of)
    except Exception:
        expl, comp = None, None       # explanation is a nicety, never fatal

    return Prediction(
        explanation=expl, comparison=comp,
        p1=name1, p2=name2, surface=surface, as_of=as_of,
        p1_win_probability=round(p, 4),
        fair_odds_p1=round(1 / p, 3) if p > 0 else 999.0,
        fair_odds_p2=round(1 / (1 - p), 3) if p < 1 else 999.0,
        confidence=confidence,
        warnings=warnings,
        freshness={
            "p1_days_since_last_match": meta["p1_days_since"],
            "p2_days_since_last_match": meta["p2_days_since"],
            "p1_form_span": f'{meta["p1_form_from"]} to {meta["p1_form_to"]}'
                            if meta["p1_form_from"] else None,
            "p2_form_span": f'{meta["p2_form_from"]} to {meta["p2_form_to"]}'
                            if meta["p2_form_from"] else None,
            "p1_rank": r1, "p2_rank": r2,
            "p1_elo": meta.get("p1_elo"), "p2_elo": meta.get("p2_elo"),
            "p1_elo_surface": meta.get("p1_elo_surface"),
            "p2_elo_surface": meta.get("p2_elo_surface"),
            "serve_stats_available": meta.get("serve_stats_available"),
        },
    )


@app.get("/predict", response_model=Prediction)
def predict(
    p1: str = Query(..., description="player name or id"),
    p2: str = Query(..., description="player name or id"),
    surface: str | None = Query(None, description="Hard | Clay | Grass"),
    tour_level: str | None = Query(None, description="G|M|A|C"),
    as_of: str | None = Query(None, description="ISO date; defaults to today"),
    best_of: int | None = Query(None, description="3 or 5"),
    court: str | None = Query(None, description="Indoor or Outdoor"),
):
    return _predict_core(p1, p2, surface, tour_level, as_of, best_of, court)


@app.get("/edge")
def edge(
    p1: str, p2: str, odds_p1: float, odds_p2: float,
    surface: str | None = None, tour_level: str | None = None,
    bankroll: float = 1000.0,
):
    """Compare the model against a quoted price and size a quarter-Kelly stake."""
    from model.backtest import KELLY_FRACTION, devig_two_way, kelly_stake

    pred = _predict_core(p1, p2, surface, tour_level)
    p = pred.p1_win_probability
    fair1, fair2 = devig_two_way(odds_p1, odds_p2)

    out = []
    for label, prob, odds, market_fair in (
        (pred.p1, p, odds_p1, fair1), (pred.p2, 1 - p, odds_p2, fair2)
    ):
        stake = kelly_stake(prob, odds)
        out.append({
            "player": label,
            "model_probability": round(prob, 4),
            "market_implied_fair": round(market_fair, 4),
            "edge": round(prob - market_fair, 4),
            "recommended_stake": round(stake * bankroll, 2),
            "stake_pct_of_bank": round(stake * 100, 2),
        })

    return {
        "sides": out,
        "confidence": pred.confidence,
        "warnings": pred.warnings + (
            ["confidence is not high; consider skipping"] if pred.confidence != "high" else []
        ),
        "kelly_fraction": KELLY_FRACTION,
    }
