# Tennis Match Prediction Pipeline

I bet on tennis and used to work through matchups by hand: recent form, opponent
quality, surface splits, head to head. I wanted to automate that, and more
importantly find out whether it actually works. Can a model built on public match
data find value against sportsbook prices, or is all of that already priced in?

The answer is no. The market is better than my model. But the probabilities it
produces are well calibrated, and getting to a trustworthy "no" meant building
the pipeline carefully enough that the result means something.

This is the third version. The first tried to cover multiple sports, but
basketball and the others each needed their own data model and their own idea of
what a matchup even is, and keeping that general made everything worse, so I cut
it back to tennis. The second was tennis only but ran on far fewer data points
and weighted them differently, leaning on factors I later found mattered less
than I thought, and it fell apart on certain matchups without any clear reason
why. That is most of the reason this version is built around calibration and
backtesting rather than just picking a side. Both earlier versions lived as local
scripts and spreadsheets, so the commit history here only covers the rewrite.

## Results

Walk-forward backtest, 11 retraining windows, 11,817 predictions on ATP/WTA
matches from Jan 2024 to Aug 2026. Trained on about 24,000 matches.

Well calibrated, but does not beat the closing line.

| | Model | Market (vig removed) |
|---|---|---|
| Brier | 0.2261 | 0.2054 |
| Accuracy | 62.7% | 67.3% |
| AUC | 0.681 | 0.742 |

Calibration is the part that worked. When the model says 60%, that player wins
about 60% of the time:

| Predicted range | n | Avg predicted | Actually won |
|---|---|---|---|
| 0.4-0.5 | 3006 | 0.457 | 0.462 |
| 0.5-0.6 | 3037 | 0.546 | 0.559 |
| 0.6-0.7 | 1872 | 0.638 | 0.638 |
| 0.7-0.8 | 866 | 0.742 | 0.718 |

ROI is negative everywhere with a real sample size: -6.6% at ATP/WTA 250-500,
-13.3% at Grand Slams, +0.1% at Masters.

The most useful thing I found: raising the edge threshold from 4% to 10% made ROI
*worse*, not better. If the model had real signal, the matches where it disagrees
most with the market should be its best bets. They're its worst instead, which
tells me those disagreements are my model being wrong rather than the market
being wrong.

Full output in `results/results_2026-08-13.txt`.

## Baselines

Same held-out matches (n=1,969, Mar to Aug 2026), same metrics.

| Predictor | Accuracy | AUC | Brier |
|---|---|---|---|
| Coin flip | 0.499 | 0.500 | 0.2500 |
| Higher rank wins | 0.618 | 0.673 | 0.2299 |
| Elo | 0.622 | 0.687 | 0.2276 |
| Surface Elo | 0.634 | 0.683 | 0.2292 |
| **Full model (35 features)** | 0.627 | 0.681 | **0.2261** |
| Closing odds (vig removed) | 0.673 | 0.742 | 0.2054 |

The model beats every simple baseline, but barely. It's 0.004 Brier over rank
alone and effectively tied with plain Elo. Most of what's predictable from public
tennis data is already in the ranking.

This harness also caught a real bug. Rank-only scored exactly 0.500 AUC, which is
impossible for a predictor with any signal. The rankings table was empty for this
data source, so both players fell back to the same default and the ranking feature
had been constant zero on every real match. A feature that never varies is a
feature that isn't working, and there's a test for it now.

## Market efficiency tests

Two scripts in `analysis/` that test the market directly. No model involved.

`bias_test.py` buckets every side by the market's de-vigged probability and checks
how often each bucket actually won. Favorites: implied 0.677, actual 0.677 over
24,106 observations. Underdogs: implied 0.322, actual 0.322. No favorite-longshot
bias at tour level.

`underdog_test.py` tests whether the market anchors on ranking and misses
fundamentals. For every match it checks whether the underdog actually had better
last-10 form, better opponent quality, and a better surface record than the
favorite. Where all three were true (n=782), the market implied 0.417 and they
won 0.416. The control group landed near zero too. The market prices form and
opponent quality correctly.

Three independent tests, same answer.

## Quick start

Runs with no API key and no downloads:

```bash
pip install -r requirements.txt
make demo     # synthetic data -> train -> backtest
make test
make serve    # API on :8000, interactive docs at /docs
```

`make demo` generates fake matches where I control each player's hidden skill
level, so the model should be able to recover it. If it can't, the bug is in my
code and not in the data.

To reproduce the real numbers, download the yearly spreadsheets from
tennis-data.co.uk into `data/`, then:

```bash
python cli.py init
python cli.py load-tennisdata data/
python cli.py elo
python cli.py train
python cli.py backtest
python cli.py baselines
```

## Data

24,053 ATP and WTA matches from 2022 to 2026, from tennis-data.co.uk, which
includes Pinnacle closing odds. Tour level only, no Challengers.

I originally planned to use Jeff Sackmann's `tennis_atp` and `tennis_wta` repos,
but they're no longer on GitHub. As of Aug 2026 his account only has
`tennis_MatchChartingProject`. `cli.py load-sackmann` still exists for anyone with
an old local clone, since that data is static and still valid.

`api-tennis.com` is wired up as a daily freshness layer but isn't running. The
dataset ends 2026-08-03, so it's missing the Cincinnati swing. The `/health`
endpoint reports this instead of hiding it.

The spreadsheets aren't in this repo. Tennis-Data holds copyright on them, so
download them yourself into `data/`.

## Architecture

```
  sources              ingest                   model            serving
  +------------+      +------------+      +------------+    +----------+
  | tennis-data|----->| adapters   |      | features   |    | FastAPI  |
  | (backfill) |      | crosswalk  |----->| (as_of)    |--->| /predict |
  | api-tennis |----->| upsert     |      | train+cal  |    | /edge    |
  | (daily)    |      | freshness  |      | backtest   |    | /health  |
  +------------+      +-----+------+      +------------+    +----------+
                            |
                            v
                     SQLite: matches, rankings,
                     players, crosswalk, odds
```

Data sources go through an adapter interface, so switching providers means
writing one class instead of rewriting the pipeline.

There's also a browser interface at `/`, which takes two player names and returns
the model's probability against the market's, plus which features drove the call.

## Things I had to get right

**No lookahead.** Every feature function takes an `as_of` date and only looks at
matches before it. Rankings are stored with valid-from and valid-to ranges instead
of being overwritten each week. Each match row also stores both players' ranks at
the time they played, so opponent-quality features can't leak. This is the easiest
way to get a fake result. If your features can see the future, your backtest looks
great and means nothing.

**Opponent quality.** My first version was `sum(weight x wins) / sum(weight)`,
which looks fine and is broken. For an all-wins record the weight cancels out, so
beating a top-10 player and beating a #300 both score exactly 1.0. I switched to
adding pseudo-counts to a Beta prior, so 5-0 over top-10s reads 0.86, 5-0 over
#300s reads 0.64, and a 2-0 record stays near 0.5 because two matches isn't
evidence.

**Randomized player order.** If player 1 is always the winner, the label is
trivially predictable and the model learns nothing real. Each match is emitted
once with a coin flip on which player goes first. There's a test asserting the
difference features flip sign when you swap the players.

**Calibration.** Raw gradient boosting scores aren't probabilities, and you can't
compute expected value from a number that isn't one. Calibration happens on a
held-out chronological slice. I clip outputs to [0.02, 0.98] because isotonic
regression will otherwise return exactly 0.0 and 1.0, which is never a defensible
thing to say about a tennis match and would mean infinite confidence in the Kelly
sizing.

**Odds are the benchmark, not a feature.** They live in a separate table. Training
on market prices would just teach the model to copy the market, which guarantees
never beating it.

**Walk-forward, not random splits.** Random k-fold on time series data leaks the
future into training.

## Commands

```bash
python cli.py init                        # create schema
python cli.py synth                       # synthetic data, no network needed
python cli.py load-tennisdata data/       # backfill
python cli.py elo                         # rebuild Elo and surface Elo
python cli.py train                       # train + calibrate
python cli.py backtest                    # walk-forward vs closing odds
python cli.py baselines                   # compare against simple predictors
python cli.py check                       # freshness / SLO report
python cli.py review                      # player names needing manual review
python cli.py ingest --days 3             # daily pull (needs API key)
```

The daily pull re-fetches 3 days, not just yesterday, because results get amended
after they're first published: late finishes, retirements, walkovers. Idempotent
upserts mean re-pulling costs nothing and catches the corrections.

## Tests

```
tests/test_pipeline.py   27 tests: idempotency, name matching, point-in-time ranks
tests/test_model.py      34 tests: leakage, symmetry, calibration bounds, dead features, Kelly
```

These target bugs that don't crash, the ones that quietly give you a good-looking
wrong answer. Three got caught this way. My synthetic player names were
`Player A001`, `Player B002`, and so on, and since name normalization strips
digits they all collapsed to one player ID, so every match was a player against
himself (zero-variance features, AUC 0.485). The second was the opponent-quality
bug above. The third was the dead ranking feature the baselines harness exposed.

## Known limitations

- Feature building runs one SQL query per player per match. Fine at this scale,
  would need caching if the dataset grew.
- `guess_tour_level` is a string heuristic on tournament names. Should be a proper
  tournament reference table.
- No serve stats. Tennis-Data doesn't carry them. The schema has slots for them
  and leaves them null rather than filling in zeros.
- SQLite is single writer. Fine for one nightly job, would need Postgres for
  concurrent ingestion.
- Best-of-five is flagged as a feature but not modeled separately, which likely
  explains why Grand Slams are the worst tour level.

## Next

- Turn on the daily API ingest so the data stays current.
- Model best-of-five separately.
- Find a data source with serve stats, since that's the largest missing signal.

## On betting

Backtested edges shrink a lot against live markets, and closing line movement
means the price you actually get usually isn't the one you modeled. This backtest
says my model doesn't beat the market, and I'm taking that seriously.
