# Setup and usage

## What you need

Python 3.10 or newer:

```bash
python --version
```

On Windows if that doesn't work, try `py --version` and use `py` instead of
`python` everywhere below. If you don't have Python, get it from python.org and
check "Add Python to PATH" on the first screen of the installer. I skipped that
the first time and nothing worked.

You also need git. Check with `git --version`. If it's missing, install it from
git-scm.com and then close your terminal and open a new one, or it still won't
find it.

---

## Part 1: just get it running

This doesn't need any data downloads. Takes about two minutes.

```bash
git clone https://github.com/apent06/tennis-ev.git
cd tennis-ev
pip install -r requirements.txt
```

Now run these three, one line at a time:

```bash
python cli.py synth
python cli.py elo
python cli.py train
```

`synth` makes 6,000 fake matches so you can test everything without downloading
anything. `elo` builds the ratings. `train` trains the model and prints how it
did.

Run them in that order. Elo gets written onto the match rows and training reads
it from there, so if you train first the Elo features are just empty.

Start the server:

```bash
python -m uvicorn api.main:app --reload
```

Go to http://127.0.0.1:8000. Keep that terminal open while you use it. Ctrl+C
stops it.

To run the tests, open a second terminal in the same folder:

```bash
python tests/test_pipeline.py
python tests/test_model.py
```

You should get 27/27 and 34/34.

---

## Part 2: real matches

### Get the data

Tennis-Data has one spreadsheet per season for each tour. Download them into
`data/`.

Windows:

```powershell
$years = 2022..2026
foreach ($y in $years) {
    Invoke-WebRequest "http://www.tennis-data.co.uk/$y/$y.xlsx"  -OutFile "data\$y.xlsx"
    Invoke-WebRequest "http://www.tennis-data.co.uk/${y}w/$y.xlsx" -OutFile "data\w$y.xlsx"
}
```

Mac or Linux:

```bash
for y in 2022 2023 2024 2025 2026; do
  curl -o "data/$y.xlsx"  "http://www.tennis-data.co.uk/$y/$y.xlsx"
  curl -o "data/w$y.xlsx" "http://www.tennis-data.co.uk/${y}w/$y.xlsx"
done
```

You need the `w` on the WTA files. Both tours use the same filename on their
site and the loader uses that prefix to tell them apart.

The spreadsheets aren't in this repo because Tennis-Data owns them, so you have
to download your own.

### Delete the fake data

If you did Part 1 the synthetic matches are still sitting in the database and
they'll mess up your results.

Windows:

```powershell
del tennis.db, tennis.db-wal, tennis.db-shm, model.pkl
```

Mac or Linux:

```bash
rm -f tennis.db tennis.db-wal tennis.db-shm model.pkl
```

### Build it

One line at a time:

```bash
python cli.py init
python cli.py load-tennisdata data/
python cli.py elo
python cli.py train
python cli.py baselines
```

Loading takes a couple minutes. Training is the slow one because it builds
features for every single match separately.

`baselines` is the interesting one. It compares the model against a coin flip,
just picking the higher ranked player, Elo, surface Elo, and the actual
sportsbook odds, all on the same matches.

### Keeping it updated

Tennis-Data updates once a week after each tournament finishes, and two weeks
after Slams. To refresh:

```powershell
Invoke-WebRequest "http://www.tennis-data.co.uk/2026/2026.xlsx"  -OutFile "data\2026.xlsx"
Invoke-WebRequest "http://www.tennis-data.co.uk/2026w/2026.xlsx" -OutFile "data\w2026.xlsx"
```

```bash
python cli.py load-tennisdata data/
python cli.py elo
python cli.py train
```

Reloading the same file doesn't duplicate anything, it just adds whatever's new.
`python cli.py check` tells you how far behind your data is.

---

## How to use it

Go to http://127.0.0.1:8000 while the server is running.

**Picking players.** Type a last name and choose from the dropdown. Names are
stored like `Alcaraz C.` because that's how the source has them. There are no
first names in this data at all.

The dropdown shows the tour, ranking, how many matches they have and when they
last played, so you can tell people apart. Always pick from the dropdown instead
of typing the whole name, because it sends the exact player. There's an ATP
`Wong C.` and a WTA `Wong C.` and typing that by hand is ambiguous.

**Price offered.** Optional. Leave it empty if you just want the probability.
Fill in both to see my number vs the market's number, the gap, and a suggested
stake. The toggle switches between decimal odds like 2.10 and percentages
like 48.

**The result.** The scale has two markers, mine and the market's after taking
out the bookmaker's cut. The bar between them is how much we disagree. Under
that you get fair odds, the edge, days of rest, the date range of each player's
last ten matches, and their rankings.

**What moved this number.** It turns off each factor one at a time, pretending
both players are equal on it, and shows how much the number changes. Bigger bar
means that factor mattered more. Bars go left for player two and right for
player one.

**Warnings.** Actually read these. They show up when someone hasn't played
recently, when their last ten matches are spread over way too long to count as
form, or when there isn't enough history. If a player has no usable history the
model just refuses instead of guessing.

---

## Other commands

```
python cli.py check        how stale the data is
python cli.py review       player names it couldn't match automatically
python cli.py backtest     walk-forward test against closing odds
python cli.py migrate      adds new columns to an existing database
python bias_test.py        are favourites or underdogs mispriced?
python underdog_test.py    does the market miss underdogs with better form?
```

---

## Stuff that goes wrong

**`python is not recognized`**
Python isn't on your PATH. Try `py`. If that doesn't work either, reinstall
Python and check the "Add Python to PATH" box.

**`git is not recognized` right after you installed git**
Close the terminal and open a new one. It won't pick it up in a window that was
already open.

**`No module named 'api'`**
Wrong folder. You need to be in the one with `cli.py` in it.

**`No model at model.pkl`**
Run `python cli.py train`.

**pip says "externally managed environment"**
Make a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate      # Mac or Linux
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

You have to run the activate line again every time you open a new terminal.

**Everything says the data is stale**
Your data stops wherever Tennis-Data last updated. Download the current season
again and reload. Some warnings will still be there and that's correct, because
someone who genuinely hasn't played in six weeks should be flagged.

**"More than one player is called X"**
Two players have that name. It lists them with tour, rank and match count so you
can click the right one. Picking from the dropdown avoids this.

**A lower ranked player gets a weird prediction**
Check their season match count under "the numbers behind it". Tennis-Data only
covers tour level, so anyone who mostly plays Challengers will show a fraction
of their actual season. Nothing I can do about that with this data source.

**PowerShell breaks when you paste a bunch of lines**
Paste one line at a time. If you paste a block it can run them together into one
broken command.

**The page looks the same after you update it**
Hard refresh with Ctrl+Shift+R.
