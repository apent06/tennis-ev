"""
Provider adapters.

The whole point of this file: trial tiers expire and providers change their
response shape. Everything downstream depends on the NORMALIZED dict below, so
swapping providers means writing one new class, not touching the pipeline.

Normalized match dict:
    match_date        str  ISO YYYY-MM-DD
    tour              str  'ATP' | 'WTA'
    tour_level        str  'G'|'M'|'A'|'C'|'F'|'D'   (Sackmann convention)
    tournament        str
    round             str
    surface           str  'Hard'|'Clay'|'Grass'|'Carpet'|None
    winner_name       str
    loser_name        str
    winner_rank       int|None    rank AS OF the match
    loser_rank        int|None
    score             str|None
    retirement        bool
    source_event_key  str|None
    source_winner_id  str|None    provider's own player id, for the crosswalk
    source_loser_id   str|None
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from datetime import date, timedelta

import requests

SURFACE_MAP = {
    "hard": "Hard", "clay": "Clay", "grass": "Grass", "carpet": "Carpet",
    "hardcourt": "Hard", "hard court": "Hard", "indoor hard": "Hard",
    "outdoor hard": "Hard", "red clay": "Clay",
}


def norm_surface(raw: str | None) -> str | None:
    if not raw:
        return None
    return SURFACE_MAP.get(str(raw).strip().lower())


def guess_tour_level(tournament: str | None) -> str | None:
    """
    Crude but adequate mapping onto Sackmann's tour_level codes.
    Worth replacing with an explicit tournament reference table once you have
    one -- this is the kind of heuristic that quietly misclassifies edge cases.
    """
    if not tournament:
        return None
    t = tournament.lower()
    if any(s in t for s in ["australian open", "roland garros", "french open",
                            "wimbledon", "us open"]):
        return "G"
    if "masters" in t or "atp 1000" in t or "wta 1000" in t:
        return "M"
    if "challenger" in t or t.startswith("ch "):
        return "C"
    if "itf" in t or "futures" in t:
        return "S"
    if "davis cup" in t or "billie jean" in t:
        return "D"
    return "A"


class MatchSource(ABC):
    name: str

    @abstractmethod
    def fetch_day(self, day: date) -> list[dict]:
        """Return normalized match dicts for all completed matches on `day`."""

    def fetch_range(self, start: date, end: date, pause: float = 0.4) -> list[dict]:
        out: list[dict] = []
        cur = start
        while cur <= end:
            out.extend(self.fetch_day(cur))
            cur += timedelta(days=1)
            time.sleep(pause)  # be polite; also keeps you inside rate limits
        return out


class ApiTennisSource(MatchSource):
    """
    Adapter for api-tennis.com.

    NOTE: verify field names against their live docs before first run -- the
    response shape below reflects their documented get_fixtures/get_events
    payload, but providers do rename things. Fail loudly rather than silently
    mapping a missing field to None.
    """

    name = "api-tennis"
    BASE = "https://api.api-tennis.com/tennis/"

    def __init__(self, api_key: str | None = None, timeout: int = 20):
        self.api_key = api_key or os.environ.get("API_TENNIS_KEY")
        if not self.api_key:
            raise RuntimeError("Set API_TENNIS_KEY in the environment.")
        self.timeout = timeout
        self.session = requests.Session()

    def _get(self, params: dict, retries: int = 3) -> dict:
        params = {**params, "APIkey": self.api_key}
        for attempt in range(retries):
            try:
                r = self.session.get(self.BASE, params=params, timeout=self.timeout)
                if r.status_code == 429:           # rate limited -> back off
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, json.JSONDecodeError):
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return {}

    def fetch_day(self, day: date) -> list[dict]:
        iso = day.isoformat()
        payload = self._get({
            "method": "get_fixtures",
            "date_start": iso,
            "date_stop": iso,
        })
        if not payload.get("success"):
            return []
        return [m for m in (self._normalize(e, iso) for e in payload.get("result", []))
                if m is not None]

    @staticmethod
    def _normalize(e: dict, iso: str) -> dict | None:
        # Only completed singles matches. Live/scheduled rows get picked up on
        # a later run -- that's exactly why we re-pull a rolling window.
        status = str(e.get("event_status", "")).lower()
        if status not in ("finished", "final", "ended"):
            return None

        p1 = e.get("event_first_player")
        p2 = e.get("event_second_player")
        if not p1 or not p2:
            return None

        winner_flag = str(e.get("event_winner", "")).strip()
        if winner_flag == "First Player":
            w_name, l_name = p1, p2
            w_key = e.get("first_player_key")
            l_key = e.get("second_player_key")
        elif winner_flag == "Second Player":
            w_name, l_name = p2, p1
            w_key = e.get("second_player_key")
            l_key = e.get("first_player_key")
        else:
            return None  # unresolved winner -> skip, pick up on the next pull

        tournament = e.get("tournament_name")
        score = e.get("event_final_result")
        return {
            "match_date": e.get("event_date") or iso,
            "tour": "WTA" if "wta" in str(tournament).lower() else "ATP",
            "tour_level": guess_tour_level(tournament),
            "tournament": tournament,
            "round": e.get("tournament_round"),
            "surface": norm_surface(e.get("tournament_surface")),
            "winner_name": w_name,
            "loser_name": l_name,
            "winner_rank": None,   # filled from our own rankings table
            "loser_rank": None,
            "score": score,
            "retirement": "ret" in str(score).lower(),
            "source_event_key": str(e.get("event_key")) if e.get("event_key") else None,
            "source_winner_id": str(w_key) if w_key else None,
            "source_loser_id": str(l_key) if l_key else None,
        }


class FixtureSource(MatchSource):
    """Reads normalized rows from a JSON file. Used for tests and offline dev."""

    name = "fixture"

    def __init__(self, path: str):
        with open(path, encoding="utf-8-sig") as f:
            self.rows = json.load(f)

    def fetch_day(self, day: date) -> list[dict]:
        return [r for r in self.rows if r["match_date"] == day.isoformat()]
