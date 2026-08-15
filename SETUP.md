# NBA +EV SCRAPING TOOL — SETUP GUIDE
# =====================================

## STEP 1: Install Python dependencies
pip install -r requirements.txt

## STEP 2: Copy environment file
cp .env.example .env

## STEP 3: (Optional) Add your Supabase credentials to .env
# Get free Supabase account at supabase.com
# Paste your URL and anon key into .env

## STEP 4: Run the tool

# Full analysis (recommended first run)
python main.py

# Run on schedule throughout the day (best for daily use)
python main.py --schedule

# Quick run using cached player data
python main.py --quick

# Just see today's props across all platforms
python main.py --props-only

# See current injury report
python main.py --injuries

## OUTPUT FILES
# All results saved to ./output/ folder
# ev_plays_latest.csv — today's +EV plays
# rolling_averages_latest.csv — player rolling stats
# platform_props_latest.csv — all platform lines
# sharp_consensus_latest.csv — sharp book benchmark

## WHAT IT PULLS:
# From NBA.com:
#   - Full game logs for all players
#   - Rolling 5/10/20 game averages
#   - Usage rates and advanced stats
#   - Opponent defensive ratings
#   - Home/away splits
#   - Team pace data
#   - Today's schedule

# From Sleeper (free official API):
#   - Player projections
#   - Live injury report

# From PrizePicks (unofficial API):
#   - All NBA props with lines

# From Underdog Fantasy (unofficial API):
#   - All Higher/Lower props

# From HotStreak, Dabble (scraped):
#   - Available props

# From DraftKings + FanDuel (sharp benchmark):
#   - No-vig consensus true probability

## HOW EV IS CALCULATED:
# 1. Pull platform line (e.g. PrizePicks: Luka PRA Over 47.5)
# 2. Get sharp no-vig true probability from DraftKings/FanDuel
# 3. Run our stats model (weighted rolling avg + matchup + injury)
# 4. Combine: 60% sharp data + 40% model
# 5. EV% = (true_probability - 0.50) * 200
# 6. Flag anything above 3% EV
# 7. Rank by EV descending
# 8. Display top 20-30 plays

## UNDERSTANDING THE OUTPUT:
# 🔥 HIGH: 7%+ EV — highest confidence, bet 2-3 units
# ⚡ MEDIUM: 4-7% EV — good edge, bet 1.5 units  
# 📊 LOW: 3-4% EV — marginal edge, bet 1 unit

## TROUBLESHOOTING:
# "No props found" — platforms may have changed their API, check GitHub for updates
# "NBA.com timeout" — add longer delays in utils/helpers.py
# Getting blocked — increase REQUEST_DELAY in .env
