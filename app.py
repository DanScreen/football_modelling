#!/usr/bin/env python
import html
import json
import math
import os
import pickle
import threading
import time
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import requests
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from Helper import league, get_points_matrix
from Get_Odds import get_odds
from Betfair import (get_worldcup_correct_score,
                     get_premier_league_correct_score, BetfairError)
from get_data import download_match_data

MODEL_PATH = os.environ.get('MODEL_PATH', 'pl_model.pkl')
GCS_BUCKET = os.environ.get('GCS_BUCKET')
MODEL_BLOB = os.environ.get('MODEL_BLOB', 'models/pl_model.pkl')
ARCHIVE_PREFIX = os.environ.get('ARCHIVE_PREFIX', 'predictions/')
API_KEY = os.environ.get('API_KEY')

# Scheduled callers (Cloud Scheduler) authenticate with a Google OIDC token
# instead of a shared secret. Cloud Run itself can't enforce that here because
# the service must stay public for the bookmarked HTML views, so the token is
# verified in-process against the service account allowed to call.
OIDC_SERVICE_ACCOUNT = os.environ.get('OIDC_SERVICE_ACCOUNT')
OIDC_AUDIENCE = os.environ.get('OIDC_AUDIENCE')

# How often an instance may ask GCS whether a newer model has been published.
MODEL_REFRESH_TTL = float(os.environ.get('MODEL_REFRESH_TTL', '60'))
DATA_STATE_BLOB = os.environ.get('DATA_STATE_BLOB', 'state/data_fingerprint.json')
DATA_STATE_PATH = os.environ.get('DATA_STATE_PATH', '.data_fingerprint.json')

FOOTBALL_DATA_URL = 'https://www.football-data.co.uk/mmz4281/'
TRAIN_START_SEASON = 1996
DATA_LEAGUES = ('E0', 'E1', 'E2')   # E2 identifies the promoted sides - see start_next_season

SUPERBRU_MAX_GOALS = 6

# Extra-time model (knockout ties only). SuperBru scores World Cup knockouts on
# the score after 120 minutes, but Betfair CORRECT_SCORE settles on 90 minutes.
# For 90-minute draws we play out extra time as independent Poisson goals: the
# total ET goal rate comes from the correct-score market's implied expected
# goals scaled to the 30-minute period (with a mild dampening for the lower
# tempo of extra time), and the home/away split of that rate is calibrated so
# the resulting win-skew matches the To Qualify market.
ET_TIME_FRACTION = 30.0 / 90.0
ET_DAMPENING = 0.9
ET_MAX_GOALS = 6      # per side, enough for a small-mean Poisson tail
SCORE_GRID = 9        # matches get_points_matrix dimensions (0..8)

CONVERT_NAMES = {
    'Arsenal': 'Arsenal', 'Aston Villa': 'Aston Villa',
    'AFC Bournemouth': 'Bournemouth', 'Bournemouth': 'Bournemouth',
    'Brentford': 'Brentford', 'Brighton and Hove Albion': 'Brighton',
    'Burnley': 'Burnley', 'Chelsea': 'Chelsea',
    'Coventry City': 'Coventry', 'Crystal Palace': 'Crystal Palace',
    'Everton': 'Everton', 'Fulham': 'Fulham',
    'Hull City': 'Hull', 'Ipswich Town': 'Ipswich',
    'Leeds United': 'Leeds', 'Leicester City': 'Leicester',
    'Liverpool': 'Liverpool', 'Luton Town': 'Luton',
    'Manchester City': 'Man City', 'Manchester United': 'Man United',
    'Newcastle United': 'Newcastle', 'Nottingham Forest': "Nott'm Forest",
    'Sheffield United': 'Sheffield United', 'Southampton': 'Southampton',
    'Sunderland': 'Sunderland', 'Tottenham Hotspur': 'Tottenham',
    'West Bromwich Albion': 'West Brom', 'West Ham United': 'West Ham',
    'Wolverhampton Wanderers': 'Wolves',
}

# Betfair's own short names, where they differ from the football-data.co.uk
# names the rest of the app uses. Only the mismatches need listing - most of
# Betfair's names already agree.
BETFAIR_NAMES = {
    'Man Utd': 'Man United',
    'Nottm Forest': "Nott'm Forest",
    'Nottingham Forest': "Nott'm Forest",
    'Spurs': 'Tottenham',
    'Sheff Utd': 'Sheffield United',
    'Sheff Wed': 'Sheffield Weds',
    'Wolverhampton': 'Wolves',
    'Brighton & Hove Albion': 'Brighton',
    'Newcastle Utd': 'Newcastle',
}

state = {
    'model': None,
    'model_generation': None,   # GCS object generation the cached model came from
    'checked_at': 0.0,          # monotonic clock of the last freshness check
}

_train_lock = threading.Lock()


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


def _poisson_pmf(lam, n):
    """PMF of a Poisson(lam) for k = 0..n as a list (index k)."""
    if lam <= 0:
        return [1.0] + [0.0] * n
    return [math.exp(-lam) * lam ** k / math.factorial(k) for k in range(n + 1)]


def _et_result_probs(lam_h, lam_a):
    """Given per-side extra-time goal rates, return the probability that the
    30 minutes of extra time is won by home / won by away / still level, as a
    (p_home, p_away, p_draw) tuple over independent Poisson goal counts."""
    ph = _poisson_pmf(lam_h, ET_MAX_GOALS)
    pa = _poisson_pmf(lam_a, ET_MAX_GOALS)
    p_home = p_away = p_draw = 0.0
    for dh, ph_k in enumerate(ph):
        for da, pa_k in enumerate(pa):
            joint = ph_k * pa_k
            if dh > da:
                p_home += joint
            elif dh < da:
                p_away += joint
            else:
                p_draw += joint
    return p_home, p_away, p_draw


def _expected_goals(scorelines):
    mu_home = sum(i * p for (i, j), p in scorelines.items())
    mu_away = sum(j * p for (i, j), p in scorelines.items())
    return mu_home, mu_away


def _solve_et_home_share(et_total, draw_total, target_draw_skew):
    """Find the home share s in [0, 1] of the extra-time goal rate such that the
    net win-skew generated within the drawn 90-minute mass matches
    target_draw_skew. Monotonic in s, so a bisection converges quickly.
    Returns s (clamped to [0, 1] if the target is unreachable)."""
    if et_total <= 0 or draw_total <= 0:
        return 0.5

    def skew(s):
        p_home, p_away, _ = _et_result_probs(s * et_total, (1.0 - s) * et_total)
        return draw_total * (p_home - p_away)

    lo, hi = 0.0, 1.0
    if target_draw_skew <= skew(lo):
        return 0.0
    if target_draw_skew >= skew(hi):
        return 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if skew(mid) < target_draw_skew:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _knockout_120min_distribution(scorelines, other, qualify):
    """Transform the 90-minute correct-score distribution into the distribution
    of the score after 120 minutes, which is what SuperBru scores knockouts on.

    Decisive 90-minute scores carry through unchanged; 90-minute draws play out
    extra time as independent Poisson goals. The total ET goal rate comes from
    the correct-score market's implied expected goals; the home/away split is
    calibrated so the resulting win-skew matches the To Qualify market (exact
    under a 50/50 penalty-shootout assumption). Returns (scorelines_120,
    other_120) in the same shape as the inputs, ready for the existing optimiser."""
    mu_home, mu_away = _expected_goals(scorelines)
    et_total = (mu_home + mu_away) * ET_TIME_FRACTION * ET_DAMPENING

    draws = {(i, j): p for (i, j), p in scorelines.items() if i == j}
    draw_total = sum(draws.values()) + other.get('draw', 0.0)

    # Decisive 90-minute result skew (fixed — these games end in normal time).
    h90 = sum(p for (i, j), p in scorelines.items() if i > j) + other.get('home', 0.0)
    a90 = sum(p for (i, j), p in scorelines.items() if i < j) + other.get('away', 0.0)

    # To Qualify pins the post-120 win-skew: q_home - q_away == W_home - W_away
    # (the shared 0.5*draw penalty term cancels). Solve for the ET split.
    q_diff = qualify['home'] - qualify['away']
    target_draw_skew = q_diff - (h90 - a90)
    s = _solve_et_home_share(et_total, draw_total, target_draw_skew)
    lam_h, lam_a = s * et_total, (1.0 - s) * et_total

    scorelines_120 = {}
    other_120 = {'home': other.get('home', 0.0), 'away': other.get('away', 0.0), 'draw': 0.0}

    # Decisive scores are settled at 90 minutes.
    for (i, j), p in scorelines.items():
        if i != j:
            scorelines_120[(i, j)] = scorelines_120.get((i, j), 0.0) + p

    # Play extra time out over each explicit 90-minute draw.
    ph = _poisson_pmf(lam_h, ET_MAX_GOALS)
    pa = _poisson_pmf(lam_a, ET_MAX_GOALS)
    for (i, _), p in draws.items():
        for dh, ph_k in enumerate(ph):
            for da, pa_k in enumerate(pa):
                prob = p * ph_k * pa_k
                if prob <= 0:
                    continue
                fh, fa = i + dh, i + da
                if fh < SCORE_GRID and fa < SCORE_GRID:
                    scorelines_120[(fh, fa)] = scorelines_120.get((fh, fa), 0.0) + prob
                else:  # overflow the grid — keep it as a result-only bucket
                    other_120[_result_region(fh, fa)] += prob

    # The 'any other draw' bucket has no explicit score; split it at result level.
    od = other.get('draw', 0.0)
    if od > 0:
        p_home, p_away, p_draw = _et_result_probs(lam_h, lam_a)
        other_120['home'] += od * p_home
        other_120['away'] += od * p_away
        other_120['draw'] += od * p_draw

    return scorelines_120, other_120


def _current_season(now=None):
    """Season end-year for a date: the 2026/27 season is season 2027.

    Seasons kick off in August, so July is the changeover point - anything from
    July onwards belongs to the season ending the following year."""
    now = now or datetime.now(timezone.utc)
    return now.year + 1 if now.month >= 7 else now.year


def _train_seasons():
    return list(range(TRAIN_START_SEASON, _current_season() + 1))


def _season_csv_url(season, league):
    return f'{FOOTBALL_DATA_URL}{str(season - 1)[2:]}{str(season)[2:]}/{league}.csv'


def _remote_fingerprint(seasons=None, leagues=DATA_LEAGUES):
    """Fingerprint the published CSVs without downloading them.

    A HEAD per file is enough: football-data.co.uk serves ETag, Last-Modified
    and Content-Length, and rewrites a season's file in place as results come
    in. Files that aren't published yet record their status code, so a season
    appearing for the first time registers as a change like any other.

    Files that error out are omitted rather than recorded, so a transient
    network failure doesn't masquerade as new data (the caller merges over the
    previous fingerprint, preserving what was already known)."""
    if seasons is None:
        current = _current_season()
        seasons = [current, current - 1]
    fingerprint = {}
    for season in seasons:
        for league in leagues:
            try:
                r = requests.head(_season_csv_url(season, league), timeout=20,
                                  allow_redirects=False)
            except requests.RequestException as e:
                print(f'HEAD {season}{league} failed: {e}')
                continue
            if r.status_code != 200:
                fingerprint[f'{season}{league}'] = f'absent:{r.status_code}'
            else:
                fingerprint[f'{season}{league}'] = (
                    r.headers.get('ETag')
                    or f"{r.headers.get('Last-Modified', '')}:{r.headers.get('Content-Length', '')}"
                )
    return fingerprint


_gcs_state = {'client': None}


def _gcs_client():
    """Cached storage client. Constructing one resolves credentials, which costs
    far more than the metadata call the freshness check is making, so it must
    not happen per request."""
    if _gcs_state['client'] is None:
        from google.cloud import storage
        _gcs_state['client'] = storage.Client()
    return _gcs_state['client']


def _model_blob():
    return _gcs_client().bucket(GCS_BUCKET).blob(MODEL_BLOB)


def _load_data_fingerprint():
    if GCS_BUCKET:
        try:
            blob = _gcs_client().bucket(GCS_BUCKET).blob(DATA_STATE_BLOB)
            if blob.exists():
                return json.loads(blob.download_as_bytes())
        except Exception as e:
            print(f'Fingerprint load failed: {e}')
        return {}
    if os.path.exists(DATA_STATE_PATH):
        with open(DATA_STATE_PATH) as f:
            return json.load(f)
    return {}


def _save_data_fingerprint(fingerprint):
    payload = json.dumps(fingerprint, indent=2, sort_keys=True)
    if GCS_BUCKET:
        _gcs_client().bucket(GCS_BUCKET).blob(DATA_STATE_BLOB).upload_from_string(
            payload, content_type='application/json'
        )
        return
    with open(DATA_STATE_PATH, 'w') as f:
        f.write(payload)


def _load_model():
    if GCS_BUCKET:
        try:
            blob = _model_blob()
            if blob.exists():
                blob.reload()  # populates generation
                model = pickle.loads(blob.download_as_bytes())
                state['model_generation'] = blob.generation
                print(f'Loaded model from gs://{GCS_BUCKET}/{MODEL_BLOB} '
                      f'(generation {blob.generation}, trained {_trained_at(model)})')
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
        blob = _model_blob()
        blob.metadata = {'trained_at': _trained_at(model) or ''}
        blob.upload_from_string(payload, content_type='application/octet-stream')
        # Remember what we just wrote, so the freshness check doesn't turn round
        # and re-download the model this instance already holds.
        state['model_generation'] = blob.generation
        print(f'Uploaded model to gs://{GCS_BUCKET}/{MODEL_BLOB} '
              f'(generation {blob.generation})')


def _trained_at(model):
    return getattr(model, 'trained_at', None)


def _ensure_fresh_model():
    """Adopt the model in GCS if it's newer than the one this instance holds.

    Prediction requests call this, but it's rate-limited to one check per
    MODEL_REFRESH_TTL seconds. A prediction is ~0.25ms of work while a GCS
    metadata round-trip is tens of milliseconds, so checking on every request
    would make the check cost far more than the thing it guards. Retraining
    happens at most once a day, so a minute of staleness is immaterial.

    Staleness is judged on the object's generation rather than a timestamp:
    it's monotonic per write and immune to clock skew between instances.

    Any failure is swallowed - a GCS blip must not take predictions down with
    it. The cached model stays in service and the next check tries again."""
    if not GCS_BUCKET:
        return
    now = time.monotonic()
    if now - state['checked_at'] < MODEL_REFRESH_TTL:
        return
    state['checked_at'] = now
    try:
        blob = _model_blob()
        blob.reload()  # metadata only - no download
        if blob.generation == state['model_generation']:
            return
        model = pickle.loads(blob.download_as_bytes())
        state['model'] = model
        state['model_generation'] = blob.generation
        print(f'Pulled newer model from GCS (generation {blob.generation}, '
              f'trained {_trained_at(model)})')
    except Exception as e:
        print(f'Model freshness check failed, keeping cached model: {e}')


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


def _verify_oidc(authorization: Optional[str]) -> bool:
    """True if the request carries a valid Google OIDC token for the caller we
    expect. Used by the Cloud Scheduler jobs so they don't need a shared secret.

    The token is verified here rather than by Cloud Run because the service is
    public - the HTML views are bookmarked on a phone - so the platform lets
    every request through and can't do it for us."""
    if not (OIDC_SERVICE_ACCOUNT and authorization):
        return False
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        return False
    try:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token
        claims = id_token.verify_oauth2_token(
            parts[1], google_requests.Request(), audience=OIDC_AUDIENCE or None
        )
    except Exception as e:
        print(f'OIDC verification failed: {e}')
        return False
    return bool(claims.get('email_verified')) and claims.get('email') == OIDC_SERVICE_ACCOUNT


def _require_caller(x_api_key: Optional[str], authorization: Optional[str] = None):
    """Scheduled jobs authenticate with an OIDC token, people with an API key."""
    if _verify_oidc(authorization):
        return
    _require_api_key(x_api_key)


def _retrain(seasons=None, download=False):
    """Train a model, roll it into the new season if needed, and publish it."""
    if download:
        download_match_data(year_range=range(TRAIN_START_SEASON, _current_season() + 1),
                            leagues=list(DATA_LEAGUES))
    seasons = seasons or _train_seasons()
    PL = league()
    PL.train_all(league_str='E0', league_below='E1', SEA=seasons)
    trained = list(getattr(PL, 'seasons_trained', None) or seasons)

    # Seasons with no E0 file yet were skipped. If the newest one has already
    # started in the divisions below, roll the model into it so the promoted
    # sides are predictable before the first Premier League results land.
    rolled = None
    pending = [s for s in seasons if s > max(trained)]
    if pending:
        rolled = PL.start_next_season(min(pending))

    PL.trained_at = datetime.now(timezone.utc).isoformat()
    state['model'] = PL
    _save_model(PL)
    return {
        'seasons': trained,
        'seasons_requested': seasons,
        'rolled_forward_to': getattr(PL, 'rolled_forward_to', None),
        'season_changes': {'out': rolled[0], 'in': rolled[1]} if rolled else None,
        'trained_at': PL.trained_at,
        'model_generation': state['model_generation'],
        'teams': list(PL.teams),
        'model_path': MODEL_PATH,
        'gcs_uri': f'gs://{GCS_BUCKET}/{MODEL_BLOB}' if GCS_BUCKET else None,
    }


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
    _ensure_fresh_model()
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

        # Knockout ties are scored on the 120-minute score, not the 90-minute
        # correct-score market. When we have a To Qualify market, remodel the
        # distribution to include extra time before optimising SuperBru points.
        knockout = bool(m.get('knockout') and m.get('qualify') and scorelines)
        if knockout:
            sb_scorelines, sb_other = _knockout_120min_distribution(
                scorelines, other, m['qualify'])
        else:
            sb_scorelines, sb_other = scorelines, other

        ml = max(sb_scorelines, key=sb_scorelines.get) if sb_scorelines else None
        sb = _superbru_optimal_betfair(sb_scorelines, sb_other)
        top = sorted(sb_scorelines.items(), key=lambda kv: kv[1], reverse=True)[:6]
        top_scorelines = [
            {'home': k[0], 'away': k[1], 'prob': round(v, 4)} for k, v in top
        ]
        out = {
            'home_team': home,
            'away_team': away,
            'commence_time': commence,
            'knockout': knockout,
            'score_basis': '120min' if knockout else '90min',
            'most_likely_score': {'home': ml[0], 'away': ml[1]} if ml else None,
            'superbru_optimal_score': sb,
            'probabilities': _outcome_probs(sb_scorelines, sb_other),
            'top_scorelines': top_scorelines,
            'overround': round(m['overround'], 4),
        }
        if knockout:
            out['qualify'] = {
                'home': round(m['qualify']['home'], 4),
                'away': round(m['qualify']['away'], 4),
            }
            # Also expose the naive 90-minute pick alongside the extra-time one,
            # so the difference the ET correction makes is visible.
            out['superbru_optimal_score_90min'] = _superbru_optimal_betfair(scorelines, other)
        results.append(out)
    return {'count': len(results), 'matches': results}


def _normalise_betfair_team(name):
    """Map a Betfair team name onto the football-data.co.uk name the model uses,
    so odds can be cross-referenced with /predictions/upcoming. Unknown names
    pass through unchanged - this endpoint reports the market, and does not need
    the model to have heard of the team."""
    if name in BETFAIR_NAMES:
        return BETFAIR_NAMES[name]
    if name in CONVERT_NAMES:
        return CONVERT_NAMES[name]
    return name


def _build_premier_league_odds(limit: int):
    """Upcoming Premier League fixtures priced straight off the Betfair exchange.

    Deliberately independent of the trained model: nothing here reads `state`,
    and no Poisson assumption is made. Every number is the de-vigged market
    price. League games settle on 90 minutes, so there is no extra-time
    correction to apply either."""
    try:
        matches = get_premier_league_correct_score(max_markets=limit)
    except BetfairError as e:
        raise HTTPException(502, f'Betfair error: {e}')
    if not matches:
        raise HTTPException(502, 'No Premier League correct-score markets available on Betfair.')

    results = []
    for m in matches[:limit]:
        home, away = m['match']
        scorelines = m['scorelines']
        other = m['other']
        commence = m['time'].isoformat() if m['time'] else None
        entry = {
            'home_team': _normalise_betfair_team(home),
            'away_team': _normalise_betfair_team(away),
            'betfair_home_team': home,
            'betfair_away_team': away,
            'competition': m.get('competition'),
            'commence_time': commence,
        }
        if not scorelines and not any(other.values()):
            entry['error'] = 'no correct-score liquidity on Betfair'
            results.append(entry)
            continue

        ml = max(scorelines, key=scorelines.get) if scorelines else None
        top = sorted(scorelines.items(), key=lambda kv: kv[1], reverse=True)[:6]
        entry.update({
            'most_likely_score': {'home': ml[0], 'away': ml[1]} if ml else None,
            'probabilities': _outcome_probs(scorelines, other),
            'top_scorelines': [
                {'home': k[0], 'away': k[1], 'prob': round(v, 4)} for k, v in top
            ],
            'other_scorelines': {k: round(v, 4) for k, v in other.items()},
            'superbru_optimal_score': _superbru_optimal_betfair(scorelines, other),
            'overround': round(m['overround'], 4),
        })
        results.append(entry)
    return {
        'source': 'betfair',
        'market': 'CORRECT_SCORE',
        'score_basis': '90min',
        'count': len(results),
        'matches': results,
    }


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model_loaded': state['model'] is not None,
        'model_trained_at': _trained_at(state['model']),
        'model_generation': state['model_generation'],
        'current_season': _current_season(),
    }


@app.post('/train')
def train(req: TrainRequest,
          x_api_key: Optional[str] = Header(default=None),
          authorization: Optional[str] = Header(default=None)):
    _require_caller(x_api_key, authorization)
    if not _train_lock.acquire(blocking=False):
        raise HTTPException(409, 'A retrain is already in progress')
    try:
        result = _retrain(seasons=req.seasons, download=req.download_data)
    finally:
        _train_lock.release()
    return {'status': 'trained', **result}


@app.post('/model/refresh')
def refresh_model(force: bool = Query(default=False),
                  x_api_key: Optional[str] = Header(default=None),
                  authorization: Optional[str] = Header(default=None)):
    """Retrain, but only if football-data.co.uk has published something new.

    Called daily by Cloud Scheduler. The check is a handful of HEAD requests
    against the current and previous season's CSVs; training only runs when one
    of them has changed, so an ordinary day costs almost nothing."""
    _require_caller(x_api_key, authorization)
    previous = _load_data_fingerprint()
    current = _remote_fingerprint()
    if not current:
        raise HTTPException(502, 'Could not reach football-data.co.uk to check for new data')
    changed = sorted(k for k, v in current.items() if previous.get(k) != v)

    if not changed and not force:
        return {
            'status': 'up-to-date',
            'checked': sorted(current),
            'changed': [],
            'trained_at': _trained_at(state['model']),
            'model_generation': state['model_generation'],
        }

    if not _train_lock.acquire(blocking=False):
        raise HTTPException(409, 'A retrain is already in progress')
    try:
        result = _retrain(download=True)
        # Only record the new fingerprint once training has actually succeeded,
        # so a failed run is retried tomorrow rather than silently skipped.
        _save_data_fingerprint({**previous, **current})
    finally:
        _train_lock.release()
    return {'status': 'retrained', 'checked': sorted(current), 'changed': changed, **result}


@app.post('/predict')
def predict(req: PredictRequest):
    _ensure_fresh_model()
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


@app.get('/odds/premierleague')
def premier_league_odds(limit: int = 20):
    """Live Betfair market prices for upcoming Premier League fixtures.

    Separate from /predictions/* by design: this is what the exchange thinks,
    with no input from the trained model."""
    return _build_premier_league_odds(limit)


@app.post('/predictions/archive')
def archive(
    limit: int = Query(default=20),
    x_api_key: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
):
    _require_caller(x_api_key, authorization)
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


@app.get('/predictions/worldcup/html', response_class=HTMLResponse)
def worldcup_html(limit: int = 20):
    payload = _build_worldcup_betfair(limit)
    return _render_worldcup_html(payload)


def _prob_bar(probs):
    h = probs['home_win'] * 100
    d = probs['draw'] * 100
    a = probs['away_win'] * 100
    return f"""
          <div class="bar" role="img" aria-label="Home {h:.0f}%, Draw {d:.0f}%, Away {a:.0f}%">
            <span class="seg home" style="width:{h:.2f}%"></span>
            <span class="seg draw" style="width:{d:.2f}%"></span>
            <span class="seg away" style="width:{a:.2f}%"></span>
          </div>
          <div class="bar-legend">
            <span class="lg"><i class="dot home"></i>Home {h:.0f}%</span>
            <span class="lg"><i class="dot draw"></i>Draw {d:.0f}%</span>
            <span class="lg"><i class="dot away"></i>Away {a:.0f}%</span>
          </div>"""


def _render_worldcup_html(payload):
    rows = []
    for m in payload['matches']:
        home = html.escape(str(m['home_team']))
        away = html.escape(str(m['away_team']))
        if 'error' in m:
            rows.append(f"""
        <article class="match unknown">
          <header><span class="team">{home}</span><span class="vs">v</span><span class="team">{away}</span></header>
          <p class="error">{html.escape(str(m['error']))}</p>
        </article>""")
            continue
        ml = m['most_likely_score']
        sb = m['superbru_optimal_score']
        ml_str = f"{ml['home']}–{ml['away']}" if ml else "—"
        kickoff = (m['commence_time'].replace('T', ' ')[:16] + ' UTC') if m.get('commence_time') else 'TBC'
        chips = []
        for s in m.get('top_scorelines', []):
            is_sb = s['home'] == sb['home'] and s['away'] == sb['away']
            chips.append(
                f"""<span class="chip{' sb' if is_sb else ''}">{s['home']}–{s['away']}"""
                f"""<i>{s['prob']*100:.0f}%</i></span>"""
            )
        chips_html = ''.join(chips) or '<span class="chip muted">no scoreline liquidity</span>'
        if m.get('knockout'):
            q = m.get('qualify', {})
            badge = (f"""<span class="badge ko" title="Scored on the score after 120 minutes; """
                     f"""extra time calibrated to the To Qualify market">120' · KO</span>""")
            kickoff_extra = (f""" · qualify: {home} {q.get('home', 0)*100:.0f}% / """
                             f"""{away} {q.get('away', 0)*100:.0f}%""") if q else ''
            # Knockout: show the naive 90' pick next to the recommended 120' pick.
            sb90 = m.get('superbru_optimal_score_90min', sb)
            scores_html = f"""
            <div class="score">
              <span class="label">Most likely</span>
              <span class="value">{ml_str}</span>
            </div>
            <div class="score muted-pick">
              <span class="label">SuperBru · 90'</span>
              <span class="value">{sb90['home']}–{sb90['away']}</span>
              <span class="ep muted-ep">{sb90['expected_points']:.2f} xPts</span>
            </div>
            <div class="score sb">
              <span class="label">SuperBru · 120' ✓</span>
              <span class="value">{sb['home']}–{sb['away']}</span>
              <span class="ep">{sb['expected_points']:.2f} xPts</span>
            </div>"""
        else:
            badge = ''
            kickoff_extra = ''
            scores_html = f"""
            <div class="score">
              <span class="label">Most likely</span>
              <span class="value">{ml_str}</span>
            </div>
            <div class="score sb">
              <span class="label">SuperBru pick</span>
              <span class="value">{sb['home']}–{sb['away']}</span>
              <span class="ep">{sb['expected_points']:.2f} xPts</span>
            </div>"""
        rows.append(f"""
        <article class="match">
          <header><span class="team">{home}</span><span class="vs">v</span><span class="team">{away}</span>{badge}</header>
          <p class="kickoff">{kickoff}{kickoff_extra}</p>
          {_prob_bar(m['probabilities'])}
          <div class="scores">{scores_html}
          </div>
          <div class="chips">{chips_html}</div>
        </article>""")
    generated = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    body = ''.join(rows) if rows else '<p class="meta">No World Cup correct-score markets available right now.</p>'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <title>World Cup Predictions</title>
  <style>
    :root {{
      --bg: #0b0e14; --card: #161a23; --card2: #11141b; --border: #232936;
      --text: #eef1f6; --muted: #8b95a7; --accent: #4ade80;
      --home: #38bdf8; --draw: #a78bfa; --away: #fb7185;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; padding: 0 16px 40px; color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: radial-gradient(1200px 600px at 50% -200px, #1b2333 0%, var(--bg) 60%); }}
    .wrap {{ max-width: 720px; margin: 0 auto; }}
    .hero {{ padding: 28px 4px 18px; }}
    .hero h1 {{ margin: 0; font-size: 26px; letter-spacing: -0.5px; }}
    .hero .sub {{ color: var(--muted); font-size: 13px; margin-top: 6px; }}
    .match {{ background: var(--card); border: 1px solid var(--border); border-radius: 14px;
              padding: 16px; margin-bottom: 14px; box-shadow: 0 1px 0 rgba(255,255,255,0.02) inset; }}
    .match header {{ display: flex; align-items: center; gap: 10px; font-size: 17px; font-weight: 650; }}
    .match .team {{ flex: 1; }}
    .match .team:last-child {{ text-align: right; }}
    .match .vs {{ color: var(--muted); font-weight: 500; font-size: 13px; flex: 0; }}
    .kickoff {{ color: var(--muted); font-size: 12px; margin: 6px 0 12px; }}
    .bar {{ display: flex; height: 10px; border-radius: 6px; overflow: hidden; background: var(--card2); }}
    .bar .seg {{ display: block; height: 100%; }}
    .bar .seg.home {{ background: var(--home); }}
    .bar .seg.draw {{ background: var(--draw); }}
    .bar .seg.away {{ background: var(--away); }}
    .bar-legend {{ display: flex; gap: 14px; margin: 8px 0 14px; font-size: 12px; color: var(--muted); }}
    .bar-legend .dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; }}
    .dot.home {{ background: var(--home); }}
    .dot.draw {{ background: var(--draw); }}
    .dot.away {{ background: var(--away); }}
    .scores {{ display: flex; gap: 12px; margin-bottom: 12px; }}
    .score {{ flex: 1; background: var(--card2); border: 1px solid var(--border);
              border-radius: 10px; padding: 10px 12px; }}
    .score.sb {{ border-color: var(--accent); background: linear-gradient(180deg, rgba(74,222,128,0.08), var(--card2)); }}
    .score.muted-pick {{ opacity: 0.72; }}
    .score .label {{ display: block; font-size: 10px; color: var(--muted); text-transform: uppercase;
                     letter-spacing: 0.6px; }}
    .score .value {{ display: block; font-size: 24px; font-weight: 750; margin-top: 3px; }}
    .score .ep {{ display: block; font-size: 11px; color: var(--accent); margin-top: 3px; font-weight: 600; }}
    .score .ep.muted-ep {{ color: var(--muted); }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .chip {{ font-size: 12px; background: var(--card2); border: 1px solid var(--border);
             border-radius: 999px; padding: 4px 10px; color: var(--text); }}
    .chip i {{ color: var(--muted); font-style: normal; margin-left: 6px; }}
    .chip.sb {{ border-color: var(--accent); color: var(--accent); }}
    .chip.sb i {{ color: var(--accent); }}
    .chip.muted {{ color: var(--muted); }}
    .badge {{ flex: 0; font-size: 10px; font-weight: 700; letter-spacing: 0.5px;
              padding: 3px 7px; border-radius: 999px; white-space: nowrap; }}
    .badge.ko {{ color: var(--accent); border: 1px solid var(--accent);
                 background: rgba(74,222,128,0.08); }}
    .match.unknown {{ opacity: 0.6; }}
    .error {{ color: #fbbf24; font-size: 13px; margin: 6px 0 0; }}
    .meta {{ color: var(--muted); font-size: 13px; }}
    footer {{ color: var(--muted); font-size: 11px; text-align: center; margin-top: 18px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>World Cup predictions</h1>
      <p class="sub">{payload['count']} matches · de-vigged Betfair correct-score odds · {generated}</p>
    </div>
    {body}
    <footer>Probabilities derived from Betfair Exchange best-back prices, normalised to remove the overround.</footer>
  </div>
</body>
</html>"""


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
