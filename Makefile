.PHONY: help install demo test train backtest serve ingest check clean docker

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

install:  ## install dependencies
	pip install -r requirements.txt

demo:  ## full offline demo: synthetic data -> train -> backtest (no network, no API key)
	python cli.py synth --players 120 --matches 6000
	python cli.py train
	python cli.py backtest

test:  ## run all tests
	python tests/test_pipeline.py
	python tests/test_model.py

train:  ## train on whatever is in the DB
	python cli.py train

backtest:  ## walk-forward backtest vs closing odds
	python cli.py backtest

ingest:  ## daily live pull (requires API_TENNIS_KEY)
	python cli.py ingest --days 3

check:  ## data freshness report
	python cli.py check

serve:  ## run the API on :8000
	uvicorn api.main:app --reload --port 8000

docker:  ## build the container
	docker build -t tennis-ev .

clean:
	rm -f tennis.db tennis.db-wal tennis.db-shm model.pkl
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
