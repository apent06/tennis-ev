# Analysis scripts

One-off tests, separate from the main pipeline. Run them from the project root
after building the database.

| Script | Question it answers |
|---|---|
| `bias_test.py` | Does the market misprice favorites or underdogs? |
| `underdog_test.py` | Does the market miss underdogs whose form is better? |
| `report.py` | Pretty-print a saved `backtest.txt` |

```bash
python analysis/bias_test.py
python analysis/underdog_test.py
```

Neither of the first two uses the model at all. They test the market directly,
which is why they live here rather than in `model/`.
