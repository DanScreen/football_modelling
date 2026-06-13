#!/usr/bin/env python
import json
import os
import pickle
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from Helper import league, get_points_matrix
from Get_Odds import get_odds
from Betfair import get_worldcup_correct_score, BetfairError
from get_data import download_match_data

MODEL_PATH = os.environ.get('MODEL_PATH', 'pl_model.pkl')
GCS_BUCKET = os.environ.get('GCS_BUCKET')
MODEL_BLOB = os.environ.get('MODEL_BLOB', 'models/pl_model.pkl')
ARCHIVE_PREFIX = os.environ.get('ARCHIVE_PREFIX', 'predictions/')
API_KEY = os.environ.get('API_KEY')
SUPERBRU_MAX_GOALS = 6

CONVERT_NAMES = {
    'Arsenal': 'Arsenal', 'Aston Villa': 'Aston Villa',
    'AFC Bournemouth': 'Bournemouth', 'Bournemouth': 'Bournemouth',
    'Brentford': 'Brentford', 'Brighton and Hove Albion': 'Brighton',
    'Burnley': 'Burnley', 'Chelsea': 'Chelsea',
    'Crystal Palace': 'Crystal Palace', 'Everton': 'Everton',
    'Fulham': 'Fulham', 'Ipswich Town': 'Ipswich',
    'Leeds United': 'Leeds', 'Leicester City': 'Leicester',
    'Liverpool': 'Liverpool', 'Luton Town': 'Luton',
    'Manchester City': 'Man City', 'Manchester United': 'Man United',
    'Newcastle United': 'Newcastle', 'Nottingham Forest': "Nott'm Forest",
    'Sheffield United': 'Sheffield United', 'Southampton': 'Southampton',
    'Sunderland': 'Sunderland', 'Tottenham Hotspur': 'Tottenham',
    'West Bromwich Albion': 'West Brom', 'West Ham United': 'West Ham',
    'Wolverhampton Wanderers': 'Wolves',
}

state = {'model': None}


def _to_int(v):
    return int(np.ravel(v)[0])


def _superbru_optimal(prediction_matrix):
    exp_points = np.zeros((SUPERBRU_MAX_GOALS, SUPERBRU_MAX_GOALS))
    for home in range(SUPERBRU_MAX_GOALS):
        for away in range(SUPERBRU_MAX_GOALS):
            exp_points[home, away] = float(np.sum(get_points_matrix(home, away) * prediction_matrix))
    idx = np.unravel_index(int(np.argmax(exp_points)), exp_points.shape)
    return {
        'home': int(idx[0]),
        'away': int(idx[1]),
        'expected_points': float(exp_points[idx]),
    }


def _result_region(home, away):
    return 'home' if home > away else ('draw' if home == away else 'away')


def _superbru_optimal_betfair(scorelines, other):
    """Pick the scoreline maximising expected SuperBru points, using real
    per-scoreline probabilities from Betfair. Explicit scorelines contribute
    exact/close/result points via the points matrix; the 'any other home/draw/
    away win' buckets can only ever yield result points, so they're added as a
    flat result-point term for any candidate whose result matches the bucket."""
    best = None
    for ch in range(SUPERBRU_MAX_GOALS):
        for ca in range(SUPERBRU_MAX_GOALS):
            pts = get_points_matrix(ch, ca)
            exp = 0.0
            for (i, j), p in scorelines.items():
                if i < 9 and j < 9:
                    exp += p * pts[i, j]
            exp += other.get(_result_region(ch, ca), 0.0)  # result point only
            if best is None or exp > best['expected_points']:
                best = {'home': ch, 'away': ca, 'expected_points': float(exp)}
    return best


def _outcome_probs(scorelines, other):
    probs = {'home': other.get('home', 0.0), 'draw': other.get('draw', 0.0), 'away': other.get('away', 0.0)}
    for (i, j), p in scorelines.items():
        probs[_result_region(i, j)] += p
    return {'home_win': probs['home'], 'draw': probs['draw'], 'away_win': probs['away']}


def _gcs_client():
    from google.cloud import storage
    return storage.Client()


def _load_model():
    if GCS_BUCKET:
        try:
            blob = _gcs_client().bucket(GCS_BUCKET).blob(MODEL_BLOB)
            if blob.exists():
                model = pickle.loads(blob.download_as_bytes())
                print(f'Loaded model from gs://{GCS_BUCKET}/{MODEL_BLOB}')
                return model
            print(f'No model at gs://{GCS_BUCKET}/{MODEL_BLOB}')
        except Exception as e:
            print(f'GCS model load failed: {e}')
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            print(f'Loaded model from {MODEL_PATH}')
            return pickle.load(f)
    return None


def _save_model(model):
    payload = pickle.dumps(model)
    with open(MODEL_PATH, 'wb') as f:
        f.write(payload)
    if GCS_BUCKET:
        _gcs_client().bucket(GCS_BUCKET).blob(MODEL_BLOB).upload_from_string(
            payload, content_type='application/octet-stream'
        )
        print(f'Uploaded model to gs://{GCS_BUCKET}/{MODEL_BLOB}')


def _archive_predictions(payload):
    if not GCS_BUCKET:
        return None
    date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    blob_path = f'{ARCHIVE_PREFIX}{date_str}.json'
    _gcs_client().bucket(GCS_BUCKET).blob(blob_path).upload_from_string(
        json.dumps(payload, indent=2, default=str), content_type='application/json'
    )
    return f'gs://{GCS_BUCKET}/{blob_path}'


def _require_api_key(provided: Optional[str]):
    if not API_KEY:
        return  # auth disabled
    if provided != API_KEY:
        raise HTTPException(401, 'Invalid or missing X-API-Key header')


@asynccontextmanager
async def lifespan(app: FastAPI):
    state['model'] = _load_model()
    yield


app = FastAPI(
    title='Football Modelling API',
    description='Bayesian football model — train, predict, and forecast upcoming Premier League fixtures.',
    lifespan=lifespan,
)


class TrainRequest(BaseModel):
    download_data: bool = False
    seasons: Optional[list[int]] = None


class PredictRequest(BaseModel):
    home_team: str
    away_team: str


def _format_prediction(home, away, prediction, commence_time=None):
    score = prediction['result']
    probs = prediction['outcomes']
    sb = _superbru_optimal(prediction['matrix'])
    out = {
        'home_team': home,
        'away_team': away,
        'most_likely_score': {'home': _to_int(score[0]), 'away': _to_int(score[1])},
        'superbru_optimal_score': {
            'home': sb['home'],
            'away': sb['away'],
            'expected_points': sb['expected_points'],
        },
        'probabilities': {
            'home_win': float(probs[0]),
            'draw': float(probs[1]),
            'away_win': float(probs[2]),
        },
    }
    if commence_time is not None:
        out['commence_time'] = commence_time.isoformat()
    return out


def _build_upcoming(limit: int):
    PL = state['model']
    if PL is None:
        raise HTTPException(503, 'Model not trained. POST /train first.')
    matches = get_odds('soccer_epl')
    if not matches:
        raise HTTPException(502, 'No upcoming matches returned by odds API.')
    results = []
    for match in matches[:limit]:
        api_home, api_away = match['match']
        home = CONVERT_NAMES.get(api_home, api_home)
        away = CONVERT_NAMES.get(api_away, api_away)
        if home not in PL.teams or away not in PL.teams:
            results.append({
                'home_team': api_home,
                'away_team': api_away,
                'commence_time': match['time'].isoformat(),
                'error': 'team not in trained model',
            })
            continue
        prediction = PL.predict(home, away)
        results.append(_format_prediction(home, away, prediction, commence_time=match['time']))
    return {'count': len(results), 'matches': results}


def _build_worldcup_betfair(limit: int):
    """Predict World Cup scorelines from real Betfair CORRECT_SCORE odds.
    No trained model and no Poisson assumption — probabilities come straight
    from the (de-vigged) exchange prices for each scoreline."""
    try:
        matches = get_worldcup_correct_score(max_markets=limit)
    except BetfairError as e:
        raise HTTPException(502, f'Betfair error: {e}')
    if not matches:
        raise HTTPException(502, 'No FIFA World Cup correct-score markets available on Betfair.')
    results = []
    for m in matches[:limit]:
        home, away = m['match']
        scorelines = m['scorelines']
        other = m['other']
        commence = m['time'].isoformat() if m['time'] else None
        if not scorelines and not any(other.values()):
            results.append({
                'home_team': home,
                'away_team': away,
                'commence_time': commence,
                'error': 'no correct-score liquidity on Betfair',
            })
            continue
        ml = max(scorelines, key=scorelines.get) if scorelines else None
        sb = _superbru_optimal_betfair(scorelines, other)
        out = {
            'home_team': home,
            'away_team': away,
            'commence_time': commence,
            'most_likely_score': {'home': ml[0], 'away': ml[1]} if ml else None,
            'superbru_optimal_score': sb,
            'probabilities': _outcome_probs(scorelines, other),
            'overround': round(m['overround'], 4),
        }
        results.append(out)
    return {'count': len(results), 'matches': results}


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': state['model'] is not None}


@app.post('/train')
def train(req: TrainRequest, x_api_key: Optional[str] = Header(default=None)):
    _require_api_key(x_api_key)
    if req.download_data:
        download_match_data(year_range=range(1996, 2027), leagues=['E0', 'E1'])
    seasons = req.seasons or list(range(1996, 2027))
    PL = league()
    PL.train_all(league_str='E0', league_below='E1', SEA=seasons)
    state['model'] = PL
    _save_model(PL)
    return {
        'status': 'trained',
        'seasons': seasons,
        'teams': list(PL.teams),
        'model_path': MODEL_PATH,
        'gcs_uri': f'gs://{GCS_BUCKET}/{MODEL_BLOB}' if GCS_BUCKET else None,
    }


@app.post('/predict')
def predict(req: PredictRequest):
    PL = state['model']
    if PL is None:
        raise HTTPException(503, 'Model not trained. POST /train first.')
    if req.home_team not in PL.teams:
        raise HTTPException(400, f"Unknown home team '{req.home_team}'. Known teams: {list(PL.teams)}")
    if req.away_team not in PL.teams:
        raise HTTPException(400, f"Unknown away team '{req.away_team}'. Known teams: {list(PL.teams)}")
    prediction = PL.predict(req.home_team, req.away_team)
    return _format_prediction(req.home_team, req.away_team, prediction)


@app.get('/predictions/upcoming')
def upcoming(limit: int = 20):
    return _build_upcoming(limit)


@app.get('/predictions/worldcup')
def worldcup(limit: int = 20):
    return _build_worldcup_betfair(limit)


@app.post('/predictions/archive')
def archive(
    limit: int = Query(default=20),
    x_api_key: Optional[str] = Header(default=None),
):
    _require_api_key(x_api_key)
    payload = _build_upcoming(limit)
    payload['generated_at'] = datetime.now(timezone.utc).isoformat()
    archived_uri = _archive_predictions(payload)
    return {'archived_uri': archived_uri, **payload}


@app.get('/predictions/today', response_class=HTMLResponse)
def today(limit: int = 20):
    payload = _build_upcoming(limit)
    return _render_html(payload)


def _render_html(payload):
    rows = []
    for m in payload['matches']:
        if 'error' in m:
            rows.append(f"""
            <article class="match unknown">
              <header>{m['home_team']} vs {m['away_team']}</header>
              <p class="error">{m['error']}</p>
            </article>""")
            continue
        ml = m['most_likely_score']
        sb = m['superbru_optimal_score']
        probs = m['probabilities']
        kickoff = m['commence_time'].replace('T', ' ')[:16] + ' UTC'
        rows.append(f"""
        <article class="match">
          <header>{m['home_team']} <span class="vs">vs</span> {m['away_team']}</header>
          <p class="kickoff">{kickoff}</p>
          <div class="scores">
            <div class="score">
              <span class="label">Most likely</span>
              <span class="value">{ml['home']}–{ml['away']}</span>
            </div>
            <div class="score sb">
              <span class="label">SuperBru pick</span>
              <span class="value">{sb['home']}–{sb['away']}</span>
              <span class="ep">{sb['expected_points']:.2f} xPts</span>
            </div>
          </div>
          <div class="probs">
            <span>H {probs['home_win']*100:.0f}%</span>
            <span>D {probs['draw']*100:.0f}%</span>
            <span>A {probs['away_win']*100:.0f}%</span>
          </div>
        </article>""")
    generated = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <title>EPL Predictions</title>
  <style>
    :root {{
      --bg: #0f1115; --card: #1a1d24; --border: #262a33;
      --text: #e8eaed; --muted: #8a93a3; --accent: #4ade80; --warn: #fbbf24;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 16px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           background: var(--bg); color: var(--text); }}
    h1 {{ margin: 8px 0 4px; font-size: 20px; }}
    .meta {{ color: var(--muted); font-size: 13px; margin-bottom: 16px; }}
    .match {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px;
              padding: 14px; margin-bottom: 12px; }}
    .match header {{ font-weight: 600; font-size: 16px; }}
    .match .vs {{ color: var(--muted); font-weight: 400; padding: 0 4px; }}
    .kickoff {{ color: var(--muted); font-size: 12px; margin: 4px 0 10px; }}
    .scores {{ display: flex; gap: 12px; margin-bottom: 10px; }}
    .score {{ flex: 1; background: #14171d; border: 1px solid var(--border);
              border-radius: 8px; padding: 8px 10px; }}
    .score.sb {{ border-color: var(--accent); }}
    .score .label {{ display: block; font-size: 11px; color: var(--muted); text-transform: uppercase;
                     letter-spacing: 0.5px; }}
    .score .value {{ display: block; font-size: 22px; font-weight: 700; margin-top: 2px; }}
    .score .ep {{ display: block; font-size: 11px; color: var(--accent); margin-top: 2px; }}
    .probs {{ display: flex; gap: 12px; font-size: 12px; color: var(--muted); }}
    .match.unknown {{ opacity: 0.6; }}
    .error {{ color: var(--warn); font-size: 13px; margin: 4px 0 0; }}
  </style>
</head>
<body>
  <h1>EPL predictions</h1>
  <p class="meta">{payload['count']} fixtures · generated {generated}</p>
  {''.join(rows) if rows else '<p class="meta">No fixtures returned.</p>'}
</body>
</html>"""


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
