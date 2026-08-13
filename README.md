# Tennis EV Analytics Platform

Match-outcome probability model for ATP/WTA tennis, with a data pipeline built
around freshness and point-in-time correctness, and an honest evaluation against
closing market odds.

The project's thesis is not "this beats the market." It's that a prediction
system is only as trustworthy as its evaluation, so every number it reports is
measured against a real baseline and every prediction carries the freshness of
the data behind it.

## Quick start (no network, no API key)

```bash
pip install -r requirements.txt
make demo          # synthetic data -> train -> walk-forward backtest
make test          # 55 tests
make serve         # API on :8000, docs at /docs
```

`make demo` generates a synthetic dataset with known latent player skill, so the
model should recover signal. If it can't, the bug is in the code, not the data.

## Architecture

```
  sources                ingest                    model              serving
  ┌──────────────┐      ┌──────────────┐      ┌─────────────┐    ┌──────────┐
  │ api-tennis   │─────▶│ adapters     │      │ features    │    │ FastAPI  │
  │ (daily)      │      │ crosswalk    │─────▶│ (as_of)     │───▶│ /predict │
  │ tennis-data  │─────▶│ upsert       │      │ train+calib │    │ /edge    │
  │ (backfill)   │      │ freshness    │      │ backtest    │    │ /health  │
  └──────────────┘      └──────┬───────┘      └─────────────┘    └──────────┘
                               ▼
                        SQLite: matches, rankings,
                        players, crosswalk, odds
```

## Data sources

**Jeff Sackmann's `tennis_atp` / `tennis_wta` repos are no longer on GitHub**
(as of Aug 2026 his account lists only `tennis_MatchChartingProject`). If you
have an old local clone the data is static and still fine — `cli.py
load-sackmann` reads it. Otherwise:

| Source | Role | Cost |
|---|---|---|
| tennis-data.co.uk | historical backfill + closing odds | free |
| api-tennis.com | daily freshness layer | trial, then paid |
| local Sackmann clone | historical backfill (if you have one) | — |
| synthetic generator | offline dev and testing | free |

Sources go through an adapter interface, so swapping providers means one new
class, not a pipeline rewrite. Verify `TD_COLUMNS` and `ApiTennisSource._normalize`
against real files before the first run — providers do rename columns.

## Design decisions

**Point-in-time correctness everywhere.** Every feature function takes `as_of`
and filters `match_date < as_of`. Rankings are versioned with
`valid_from`/`valid_to` and never updated in place. `winner_rank`/`loser_rank`
are stored on the match row, so opponent-quality features are leak-free with no
join. Without this a backtest sees the future and reports an edge that isn't there.

**Opponent quality via Beta shrinkage, not a weighted ratio.** The obvious
implementation — `sum(weight × won) / sum(weight)` — is silently broken: for an
all-wins record the weight cancels, so beating a top-10 and beating a #300 both
score exactly 1.0. Instead each result adds pseudo-counts to a Beta prior, so
5-0 over top-10s reads 0.86, 5-0 over #300s reads 0.64, and thin records stay
near 0.5.

**Randomized player order in training.** If p1 is always the winner the label is
trivially predictable. Each match is emitted once with a coin flip on which
player is p1. A test asserts the diff features are antisymmetric under swap.

**Calibration is not optional.** Raw gradient-boosting scores are not
probabilities, and EV sizing needs probabilities. Calibration happens on a
held-out chronological slice, with method chosen by sample size — isotonic needs
data to be smooth, Platt is better-behaved when thin. Outputs are clipped to
[0.02, 0.98]: isotonic will otherwise emit exactly 0.0 and 1.0, which is never a
defensible claim about a tennis match and implies infinite Kelly confidence.

**Staleness travels with the prediction.** `/predict` never returns a bare
number. Every response carries days-since-last-match for both players and a
confidence label that degrades on stale or thin data. If a player has no usable
history the endpoint returns 422 rather than a confident number built on
defaults.

**Closing odds are a benchmark, not a feature.** They live in a separate table.
Training on market prices teaches the model to imitate the market, which
guarantees never beating it.

**Walk-forward, never random k-fold.** Random splits on time-series data leak
the future into training.

## What the backtest reports

- calibration table (predicted vs observed frequency per bucket)
- Brier and log loss against the market's own de-vigged probabilities
- ROI at a given edge threshold, quarter-Kelly, capped at 2% of bank
- the same broken out **by tour level** — if an edge exists it's usually at
  Challenger level, where fewer sharp bettors are looking

Closing odds minus vig is a hard baseline; most models that reproduce public
data land at or below it. Finding that out precisely is the point. A result like
"+1.2% at Challenger level, negative on tour" is a stronger claim than an
unqualified profit number, because it's falsifiable.

## Commands

```bash
python cli.py init                          # create schema
python cli.py synth                         # synthetic data, no network
python cli.py load-tennisdata data/td/      # backfill from tennis-data.co.uk
python cli.py load-sackmann data/sackmann/  # backfill from a local clone
python cli.py ingest --days 3               # daily live pull
python cli.py train                         # train + calibrate
python cli.py backtest                      # walk-forward vs closing odds
python cli.py check                         # freshness / SLO report
python cli.py review                        # unresolved player names
```

Nightly: `0 5 * * * cd /path && python cli.py ingest --days 3`

The 3-day rolling window is deliberate — results get amended after first
publication (late finishes, retirements, walkovers), and idempotent upserts
catch those for free.

## Testing

```
tests/test_pipeline.py   27 tests — idempotency, crosswalk, point-in-time ranks
tests/test_model.py      28 tests — leakage, symmetry, calibration bounds, Kelly
```

The tests target failures that don't raise exceptions — the ones that quietly
produce a good-looking number that's wrong. Two real bugs were caught this way
during development: a name-normalization collision that made every synthetic
player resolve to one id (zero-variance features, AUC 0.485), and the
weighted-ratio quality score described above.

## Known limitations

- Feature building is one SQL query per player per match. Fine for tens of
  thousands of matches; cache the matrix if it grows.
- `guess_tour_level` is a string heuristic. Replace with a tournament reference
  table when you have one.
- No indoor/outdoor distinction. tennis-data.co.uk has a `Court` column if you
  want it.
- SQLite is single-writer. Fine for one nightly job; move to Postgres for
  concurrent ingestion.
- Synthetic ROI figures are meaningless — tour level is assigned at random there
  and unrelated to skill. Only real data gives a meaningful betting result.

## Responsible use

Backtested edges shrink substantially against live markets, and closing-line
movement means the price you get is rarely the price you modeled. If you bet on
this, size to what you'd be fine losing entirely.

## Results

Walk-forward backtest, 11 retraining windows over 11,817 ATP/WTA matches
(Jan 2024 – Aug 2026). Trained on ~24k matches from tennis-data.co.uk.

**The model is well calibrated but does not beat the closing line.**

| | Model | Market (de-vigged) |
|---|---|---|
| Brier | 0.2283 | 0.2058 |
| Log loss | 0.6478 | 0.5965 |
| Accuracy | 61.8% | — |

Calibration is sound across the range — predicted probability tracks observed
frequency closely in every high-volume bucket:

| Predicted | n | Predicted | Observed |
|---|---|---|---|
| 0.4–0.5 | 3006 | 0.457 | 0.462 |
| 0.5–0.6 | 3037 | 0.546 | 0.559 |
| 0.6–0.7 | 1872 | 0.638 | 0.638 |
| 0.7–0.8 | 866 | 0.742 | 0.718 |

ROI is negative at every tour level with a meaningful sample (−6.6% at ATP/WTA
250-500, −13.3% at Grand Slams, +0.1% at Masters). Raising the edge threshold
from 4% to 10% made ROI *worse*, not better — if the model had exploitable
signal, its largest disagreements with the market should be its best bets.
Instead they're its worst, which indicates those gaps reflect model error rather
than market inefficiency.

This is the expected outcome for a model built on public match data, and
finding it out precisely was the point. The calibration result is the useful
one: the probabilities are trustworthy even though the edge isn't there.
