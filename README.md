# Football Modelling API

A FastAPI service that exposes a Bayesian football model for training and match prediction.
The underlying model uses Gamma–Poisson conjugate priors over team attacking/defensive
strengths and home-ground advantage, trained on historical results from
[football-data.co.uk](https://www.football-data.co.uk/).

## Project layout

| File | Purpose |
|---|---|
| `app.py` | FastAPI application — train, predict, and forecast upcoming fixtures |
| `Helper.py` | Core Bayesian model classes (`league`, `league_fast`) and CSV parsers |
| `get_data.py` | Download match/betting CSVs (`fetch_csv`, `download_match_data`, `download_betting_data`) |
| `Get_Odds.py` | Wrapper around [the-odds-api.com](https://the-odds-api.com/) for upcoming fixtures |
| `*.ipynb` | Interactive analysis notebooks (Premier League, Superbru, Profitability, Optimisation, etc.) |
| `AutoData/`, `BettingData/` | Downloaded CSVs (created on first data download) |
| `pl_model.pkl` | Pickled trained model (created on first `/train`) |

## Setup

Requires Python 3.10+.

```bash
# Create / activate the project venv (if you don't have one)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install fastapi 'uvicorn[standard]' pandas numpy scipy requests
```

Optional environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `ODDS_API_KEY` | hardcoded fallback in `Get_Odds.py` | API key for [the-odds-api.com](https://the-odds-api.com/) |

## Running the API

```bash
# Development (auto-reload)
uvicorn app:app --reload

# Or run directly
python app.py
```

The server starts on `http://localhost:8000`. Interactive Swagger docs are available at
`http://localhost:8000/docs`.

A pickled model from a previous run (`pl_model.pkl`) is auto-loaded on startup, so you do
not need to retrain on every restart.

## Endpoints

### `GET /health`

Service liveness check.

```json
{ "status": "ok", "model_loaded": true }
```

### `POST /train`

Train the Premier League model. Pickles the trained model to `pl_model.pkl`.

Request body:

```json
{
  "download_data": false,
  "seasons": null
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `download_data` | bool | `false` | If true, download fresh `E0`/`E1` CSVs into `AutoData/` before training |
| `seasons` | list[int] \| null | `null` | Season end-years to train on. Defaults to `1996…2026` |

Response: `{ "status": "trained", "seasons": [...], "teams": [...], "model_path": "pl_model.pkl" }`

Example:

```bash
curl -X POST http://localhost:8000/train \
     -H 'Content-Type: application/json' \
     -d '{"download_data": true}'
```

### `POST /predict`

Predict the score and outcome probabilities for a single match.

Request body:

```json
{ "home_team": "Arsenal", "away_team": "Chelsea" }
```

Team names must match the football-data.co.uk convention (e.g. `Man City`, `Man United`,
`Tottenham`, `Nott'm Forest`). Returns `400` if either team is not in the trained model
and `503` if no model is loaded.

Response:

```json
{
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "predicted_score": { "home": 2, "away": 1 },
  "probabilities": {
    "home_win": 0.52,
    "draw": 0.26,
    "away_win": 0.22
  }
}
```

### `GET /predictions/upcoming`

Predicted scores for the next N upcoming Premier League fixtures, pulled live from the
odds API. Defaults to 20.

Query parameters:

| Param | Type | Default | Description |
|---|---|---|---|
| `limit` | int | `20` | Maximum number of fixtures to return |

Fixtures referencing a team not in the trained model (e.g. a newly promoted side that
hasn't been included in training) are returned with an `error` field instead of a
prediction.

Example:

```bash
curl 'http://localhost:8000/predictions/upcoming?limit=10'
```

Response:

```json
{
  "count": 10,
  "matches": [
    {
      "home_team": "Liverpool",
      "away_team": "Man City",
      "commence_time": "2026-05-18T16:30:00+00:00",
      "predicted_score": { "home": 2, "away": 1 },
      "probabilities": {
        "home_win": 0.48,
        "draw": 0.27,
        "away_win": 0.25
      }
    }
  ]
}
```

## Typical workflow

1. Start the server: `uvicorn app:app --reload`
2. First-time setup: `POST /train` with `{"download_data": true}` to download CSVs and train.
3. Subsequent restarts auto-load `pl_model.pkl` — no retraining needed unless you want fresh data.
4. Hit `GET /predictions/upcoming` for fixture-by-fixture predictions.

## Notebooks

The interactive notebooks (`Premier League.ipynb`, `Superbru.ipynb`, `Run All Leagues.ipynb`,
`Profitability Of All Leagues.ipynb`, `Optimisation.ipynb`, `New Football Model.ipynb`)
all import shared logic via `%run get_data.py` and `%run Helper.py`. Use them for
exploratory analysis, profitability backtests, and Superbru predictions — the FastAPI
service is the production interface for live predictions.
