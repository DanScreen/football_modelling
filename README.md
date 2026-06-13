# Football Modelling API

A FastAPI service that exposes a Bayesian football model for training and match prediction.
The underlying model uses Gamma–Poisson conjugate priors over team attacking/defensive
strengths and home-ground advantage, trained on historical results from
[football-data.co.uk](https://www.football-data.co.uk/).

Each match prediction returns **two** scorelines:
- `most_likely_score` — the single scoreline with the highest probability under the model
- `superbru_optimal_score` — the scoreline that maximises expected SuperBru points,
  computed by weighting the SuperBru points matrix for each candidate scoreline by the
  model's full joint probability distribution over scores

## Project layout

| File | Purpose |
|---|---|
| `app.py` | FastAPI application — train, predict, forecast upcoming fixtures, render HTML, archive to GCS |
| `Helper.py` | Core Bayesian model classes (`league`, `league_fast`) and CSV parsers |
| `get_data.py` | Download match/betting CSVs (`fetch_csv`, `download_match_data`, `download_betting_data`) |
| `Get_Odds.py` | Wrapper around [the-odds-api.com](https://the-odds-api.com/) for upcoming fixtures |
| `Betfair.py` | Betfair Exchange API client — pulls real `CORRECT_SCORE` odds for World Cup fixtures |
| `*.ipynb` | Interactive analysis notebooks (Premier League, Superbru, Profitability, Optimisation, etc.) |
| `AutoData/`, `BettingData/` | Downloaded CSVs (created on first data download) |
| `pl_model.pkl` | Pickled trained model (created on first `/train`) |
| `Dockerfile` / `.dockerignore` | Container build for Cloud Run |
| `requirements.txt` | Pinned Python deps for the container |
| `terraform/` | Cloud Run, GCS, Cloud Scheduler, and Workload Identity Federation infra |
| `.github/workflows/deploy.yml` | GitHub Actions pipeline that builds and deploys to Cloud Run |

## Setup

Requires Python 3.10+.

```bash
# Create / activate the project venv (if you don't have one)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install fastapi 'uvicorn[standard]' pandas numpy scipy requests
```

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `ODDS_API_KEY` | hardcoded fallback in `Get_Odds.py` | API key for [the-odds-api.com](https://the-odds-api.com/) |
| `MODEL_PATH` | `pl_model.pkl` | Local pickle path for the trained model |
| `GCS_BUCKET` | _(unset)_ | If set, model is loaded/saved from this bucket and predictions are archived there |
| `MODEL_BLOB` | `models/pl_model.pkl` | GCS object path for the model pickle |
| `ARCHIVE_PREFIX` | `predictions/` | GCS prefix for daily prediction archive (key is `<prefix><YYYY-MM-DD>.json`) |
| `API_KEY` | _(unset, auth disabled)_ | If set, `/train` and `/predictions/archive` require `X-API-Key: <value>` |
| `BETFAIR_APP_KEY` | _(unset)_ | Betfair Delayed Application Key — required for `/predictions/worldcup` |
| `BETFAIR_USERNAME` | _(unset)_ | Betfair account username (interactive login) |
| `BETFAIR_PASSWORD` | _(unset)_ | Betfair account password (interactive login) |
| `BETFAIR_WC_QUERY` | `FIFA World Cup` | Text query used to locate the World Cup competition on the exchange |
| `PORT` | `8000` (`8080` in container) | Port uvicorn listens on |

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
  "most_likely_score": { "home": 2, "away": 1 },
  "superbru_optimal_score": { "home": 1, "away": 1, "expected_points": 1.42 },
  "probabilities": {
    "home_win": 0.52,
    "draw": 0.26,
    "away_win": 0.22
  }
}
```

`most_likely_score` is the single highest-probability scoreline. `superbru_optimal_score`
is the scoreline that maximises expected SuperBru points (1.5 for "close", 3 for exact,
plus result points), computed by integrating each candidate scoreline's points matrix
against the model's joint probability distribution. `expected_points` is the resulting
expected SuperBru score for picking that scoreline. The two scorelines often differ —
e.g. a 1-1 draw can be the SuperBru-optimal pick even when 2-1 is more likely, because
1-1 collects "close" points across more probable outcomes.

### `GET /predictions/today`

Mobile-friendly HTML page summarising the next N upcoming fixtures with most-likely
scoreline, SuperBru-optimal scoreline (with expected points), and outcome probabilities.
Designed to be bookmarked on an iPhone home screen — dark theme, no JS, single-tap viewing.

```
http://localhost:8000/predictions/today?limit=20
```

### `POST /predictions/archive`

Same data as `/predictions/upcoming`, but also writes the JSON to GCS at
`gs://$GCS_BUCKET/${ARCHIVE_PREFIX}<YYYY-MM-DD>.json` for backtesting later. Intended
to be called once a day by Cloud Scheduler. Requires `X-API-Key` header if `API_KEY` is set.

```bash
curl -X POST 'https://<service-url>/predictions/archive?limit=20' \
     -H "X-API-Key: $API_KEY"
```

Response includes `archived_uri`, `generated_at`, `count`, and `matches` (same shape as `/predictions/upcoming`).

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
      "most_likely_score": { "home": 2, "away": 1 },
      "superbru_optimal_score": { "home": 1, "away": 1, "expected_points": 1.38 },
      "probabilities": {
        "home_win": 0.48,
        "draw": 0.27,
        "away_win": 0.25
      }
    }
  ]
}
```

### `GET /predictions/worldcup`

SuperBru-optimal scorelines for upcoming **FIFA World Cup** fixtures, derived from **real
per-scoreline odds on the Betfair Exchange** — no trained model and no Poisson assumption.

How it works:

1. Logs in to the Betfair Exchange API (interactive login, free Delayed App Key).
2. Finds upcoming World Cup `CORRECT_SCORE` markets (soccer `eventTypeId=1`) and pulls the
   best-back price for every quoted scoreline runner.
3. **De-vigs** the whole correct-score ladder — including the *Any Other Home/Draw/Away Win*
   buckets — by normalising the implied probabilities back to sum to 1.
4. Picks both the most-likely explicit scoreline and the **SuperBru-optimal** scoreline,
   integrating the real scoreline probabilities against the SuperBru points matrix. The
   *Any Other …* buckets contribute result points only (they can't yield exact/close points).

This endpoint does **not** require a trained model — probabilities come straight from the
exchange. It **does** require `BETFAIR_APP_KEY`, `BETFAIR_USERNAME`, and `BETFAIR_PASSWORD`.

Query parameters:

| Param | Type | Default | Description |
|---|---|---|---|
| `limit` | int | `20` | Maximum number of fixtures to return |

```bash
curl 'http://localhost:8000/predictions/worldcup?limit=10'
```

Response:

```json
{
  "count": 1,
  "matches": [
    {
      "home_team": "Brazil",
      "away_team": "Serbia",
      "commence_time": "2026-06-20T19:00:00+00:00",
      "most_likely_score": { "home": 1, "away": 0 },
      "superbru_optimal_score": { "home": 1, "away": 0, "expected_points": 0.93 },
      "probabilities": { "home_win": 0.55, "draw": 0.33, "away_win": 0.11 },
      "overround": 1.06
    }
  ]
}
```

`probabilities` are the de-vigged exchange win/draw/win probabilities; `overround` is the raw
book sum before de-vigging (a measure of the market margin). Fixtures with no correct-score
market or liquidity are returned with an `error` field instead of a prediction.

> Notes:
> - Use the **free Delayed Application Key** — delayed prices are fine for daily predictions;
>   the paid Live key is only needed to place bets in real time.
> - The exchange typically only lists `CORRECT_SCORE` markets with liquidity close to kickoff
>   for major fixtures, so early group games may be thin or absent until nearer the tournament.

## Typical workflow

1. Start the server: `uvicorn app:app --reload`
2. First-time setup: `POST /train` with `{"download_data": true}` to download CSVs and train.
3. Subsequent restarts auto-load `pl_model.pkl` — no retraining needed unless you want fresh data.
4. Hit `GET /predictions/upcoming` for fixture-by-fixture predictions.

## Deployment to GCP Cloud Run

The service is designed to run on Cloud Run. The deployment provisions:

- **Cloud Run service** (`football-api`) running the FastAPI container
- **Artifact Registry** repo for the Docker image
- **GCS bucket** (`<project>-football-data`) holding the model pickle and daily prediction archive
- **Secret Manager** secrets for `ODDS_API_KEY`, `API_KEY`, and Betfair credentials (`betfair-app-key`, `betfair-username`, `betfair-password`)
- **Cloud Scheduler** job that POSTs `/predictions/archive` daily
- **Workload Identity Federation** so GitHub Actions can deploy without service-account JSON keys

### One-time infrastructure setup

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars: set project_id and github_repo

terraform init
terraform apply
```

Outputs include:
- `service_url` — public Cloud Run URL (bookmark `<url>/predictions/today` on your phone)
- `bucket_name`, `deployer_service_account`, `workload_identity_provider`, `artifact_registry_uri`

Populate the Secret Manager secrets with their values. Because the Cloud Run service mounts
these secrets at version `latest`, each secret must have **at least one version before the
service is deployed** — otherwise the revision fails to start. If `terraform apply` errors
on the Cloud Run service because a secret has no versions, create the secret containers
first with a targeted apply, add the versions, then re-run the full apply:

```bash
# (only if needed) create just the secret containers first
terraform apply \
  -target=google_secret_manager_secret.odds_api_key \
  -target=google_secret_manager_secret.api_key \
  -target=google_secret_manager_secret.betfair_app_key \
  -target=google_secret_manager_secret.betfair_username \
  -target=google_secret_manager_secret.betfair_password

# add the actual secret values (replace the quoted text with your real values)
echo -n 'your-odds-api-key'   | gcloud secrets versions add odds-api-key     --data-file=-
echo -n 'your-shared-api-key' | gcloud secrets versions add football-api-key --data-file=-
echo -n 'your-betfair-app-key'  | gcloud secrets versions add betfair-app-key  --data-file=-
echo -n 'your-betfair-username' | gcloud secrets versions add betfair-username --data-file=-
echo -n 'your-betfair-password' | gcloud secrets versions add betfair-password --data-file=-

# then run the full apply to wire everything into Cloud Run
terraform apply

# Patch the Cloud Scheduler job to use the real key
gcloud scheduler jobs update http football-daily-archive \
  --location <region> \
  --update-headers "X-API-Key=your-shared-api-key,Content-Type=application/json"
```

Train the model locally and upload the pickle to GCS so cold starts can load it:

```bash
python app.py &
curl -X POST localhost:8000/train -H "Content-Type: application/json" -d '{"download_data": true}'
gsutil cp pl_model.pkl gs://<project>-football-data/models/pl_model.pkl
```

(Alternatively, after the container is deployed, hit `/train` against the Cloud Run URL — it
will write the pickle directly to GCS.)

### GitHub Actions setup

Add the following to your GitHub repo:

- **Repository variables** (`Settings → Secrets and variables → Actions → Variables`):
  - `GCP_PROJECT` — your project ID
  - `GCP_REGION` — e.g. `europe-west2`
- **Repository secrets**:
  - `WIF_PROVIDER` — value of the `workload_identity_provider` Terraform output
  - `DEPLOYER_SA` — value of the `deployer_service_account` output

Pushes to `main` or `deploy_to_gcp` will build the image, push it to Artifact Registry,
and roll out a new Cloud Run revision.

### Daily flow

1. **08:00 UTC** Cloud Scheduler fires → `POST /predictions/archive` on Cloud Run.
2. Cloud Run loads the pickled model from GCS (or reuses the warm instance), calls the odds
   API for upcoming fixtures, computes predictions, and writes
   `gs://<bucket>/predictions/<YYYY-MM-DD>.json`.
3. Whenever you want to view, open the bookmark on your iPhone:
   `https://<service-url>/predictions/today` — renders the latest fixtures as a mobile
   page with most-likely and SuperBru-optimal scorelines.

## Notebooks

The interactive notebooks (`Premier League.ipynb`, `Superbru.ipynb`, `Run All Leagues.ipynb`,
`Profitability Of All Leagues.ipynb`, `Optimisation.ipynb`, `New Football Model.ipynb`)
all import shared logic via `%run get_data.py` and `%run Helper.py`. Use them for
exploratory analysis, profitability backtests, and Superbru predictions — the FastAPI
service is the production interface for live predictions.
