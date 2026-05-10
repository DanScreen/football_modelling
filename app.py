#!/usr/bin/env python
import os
import pickle
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from Helper import league
from Get_Odds import get_odds
from get_data import download_match_data

MODEL_PATH = 'pl_model.pkl'

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                state['model'] = pickle.load(f)
            print(f'Loaded model from {MODEL_PATH}')
        except Exception as e:
            print(f'Failed to load {MODEL_PATH}: {e}')
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
    out = {
        'home_team': home,
        'away_team': away,
        'predicted_score': {'home': _to_int(score[0]), 'away': _to_int(score[1])},
        'probabilities': {
            'home_win': float(probs[0]),
            'draw': float(probs[1]),
            'away_win': float(probs[2]),
        },
    }
    if commence_time is not None:
        out['commence_time'] = commence_time.isoformat()
    return out


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': state['model'] is not None}


@app.post('/train')
def train(req: TrainRequest):
    if req.download_data:
        download_match_data(year_range=range(1996, 2027), leagues=['E0', 'E1'])
    seasons = req.seasons or list(range(1996, 2027))
    PL = league()
    PL.train_all(league_str='E0', league_below='E1', SEA=seasons)
    state['model'] = PL
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(PL, f)
    return {
        'status': 'trained',
        'seasons': seasons,
        'teams': list(PL.teams),
        'model_path': MODEL_PATH,
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


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
