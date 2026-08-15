"""
Per-prediction explanation.

For this matchup, which factors pushed the number toward
which player, and by how much?

Method, leave-one-out neutralization. Take the real prediction, then re-run it
with one feature reset to "these two players are identical on this", and measure
how far the probability moves. That movement is what the feature was worth here.

    contribution_i = p_actual - p_with_feature_i_neutralized

Positive means the feature favoured player 1; negative favoured player 2.

Why not SHAP: SHAP would be more rigorous (it accounts for interactions by
averaging over feature orderings) but needs an extra dependency. Leave-one-out
is a reasonable approximation and is far easier to explain, which matters more
here than the last few decimal places.

Honest limits, worth stating wherever this is displayed:
  - Contributions won't sum exactly to the total, because gradient boosting uses
    feature interactions and this method ignores them.
  - This explains what the model did. It is not evidence the model is right --
    backtesting showed it doesn't beat closing odds.
"""

from __future__ import annotations

import numpy as np

from .features import FEATURE_LABELS, FEATURE_NAMES, NEUTRAL
from .train import predict_calibrated


def explain(model, feats: dict, top_n: int = 6) -> dict:
    """
    Attribute a single prediction.

    `feats` is the dict from build_features()["features"].
    Returns the actual probability plus a ranked list of contributions.
    """
    base_row = [feats[n] for n in FEATURE_NAMES]
    p_actual = float(predict_calibrated(model, [base_row])[0])

    # Build every neutralized variant at once, one predict call, not eighteen.
    variants, names = [], []
    for i, name in enumerate(FEATURE_NAMES):
        if name not in NEUTRAL:
            continue                      # context feature, not directional
        row = list(base_row)
        row[i] = NEUTRAL[name]
        variants.append(row)
        names.append(name)

    if not variants:
        return {"probability": p_actual, "contributions": []}

    probs = predict_calibrated(model, variants)

    contribs = []
    for name, p_without in zip(names, probs):
        delta = p_actual - float(p_without)
        if abs(delta) < 0.001:            # below a tenth of a point: noise
            continue
        contribs.append({
            "feature": name,
            "label": FEATURE_LABELS.get(name, name),
            "raw_value": round(float(feats[name]), 4),
            "delta": round(delta, 4),
            "favours": 1 if delta > 0 else 2,
        })

    contribs.sort(key=lambda c: -abs(c["delta"]))
    total = sum(abs(c["delta"]) for c in contribs)

    return {
        "probability": round(p_actual, 4),
        "contributions": contribs[:top_n],
        "n_considered": len(names),
        "explained_movement": round(total, 4),
        "note": ("Each figure is how far the probability moves if the two "
                 "players are made equal on that factor. They don't sum to the "
                 "total because the model uses interactions between features."),
    }


def side_by_side(conn, p1_id: str, p2_id: str, surface: str | None,
                 as_of: str) -> list[dict]:
    """
    The underlying numbers for each player, unprocessed.

    Separate from attribution on purpose: this is what the players actually did,
    where attribution is what the model made of it.
    """
    from .features import h2h_features, player_features

    f1 = player_features(conn, p1_id, surface, as_of)
    f2 = player_features(conn, p2_id, surface, as_of)
    h = h2h_features(conn, p1_id, p2_id, as_of)

    def pct(x):
        return f"{x * 100:.0f}%"

    rows = [
        ("Last 10", f"{f1['wins_10']}-{f1['losses_10']}" if "wins_10" in f1
         else pct(f1["form_win_rate_10"]),
         f"{f2['wins_10']}-{f2['losses_10']}" if "wins_10" in f2
         else pct(f2["form_win_rate_10"])),
        ("Last 5", pct(f1["form_win_rate_5"]), pct(f2["form_win_rate_5"])),
        ("Season win rate", pct(f1["season_win_rate"]), pct(f2["season_win_rate"])),
        ("Season matches", str(f1["season_matches"]), str(f2["season_matches"])),
        (f"{surface or 'Surface'} win rate",
         pct(f1["surface_win_rate"]) + f" ({f1['surface_n']})",
         pct(f2["surface_win_rate"]) + f" ({f2['surface_n']})"),
        ("Opponent quality (season)",
         f"{f1['quality_score_season']:.2f}", f"{f2['quality_score_season']:.2f}"),
        ("Opponent quality (recent)",
         f"{f1['quality_score_recent']:.2f}", f"{f2['quality_score_recent']:.2f}"),
        ("Matches last 14 days", str(f1["load_14d"]), str(f2["load_14d"])),
        ("Days since last match",
         str(f1["days_since_last"]) if f1["days_since_last"] < 999 else "—",
         str(f2["days_since_last"]) if f2["days_since_last"] < 999 else "—"),
    ]

    if h["h2h_n"]:
        won_by_1 = round(h["h2h_raw"] * h["h2h_n"])
        rows.append(("Head-to-head",
                     f"{won_by_1}-{h['h2h_n'] - won_by_1}",
                     f"{h['h2h_n'] - won_by_1}-{won_by_1}"))
    else:
        rows.append(("Head-to-head", "never met", "never met"))

    return [{"label": a, "p1": b, "p2": c} for a, b, c in rows]
