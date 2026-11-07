#!/usr/bin/env python
# coding: utf-8
"""Client for the Scorers bot API (scorersonline.uk).

Scorers exposes a public API under /api/v1/ so automated models can play in a
league alongside humans. Auth is a bearer API key that maps to exactly one
account, so "submitting on behalf of" a stream is simply holding that stream's
key: the Betfair market and the Bayesian model each get their own account, and
each therefore gets its own row on the leaderboard.

See ../FootballPredictionWebsite/docs/api.md for the endpoint contract.
"""

import os
import re
import difflib
import datetime

import requests

DEFAULT_BASE_URL = 'https://scorersonline.uk'

# Team names reach us in four different vocabularies - football-data.co.uk (the
# model), Betfair (the odds), football-data.org shortName (what Scorers imports)
# and the occasional full club name. Rather than maintain a fourth mapping
# table, fixtures are matched on kickoff time plus a fuzzy name comparison over
# a canonical form, and an unconfident match is skipped rather than guessed.
_NOISE = re.compile(r'\b(fc|afc|cf|club|the|and|hove|albion|wanderers|hotspur)\b')
_APOSTROPHE = re.compile(r"[''`]")
_NON_ALNUM = re.compile(r'[^a-z0-9 ]+')

# Deliberately NOT stripped as noise: 'city', 'united' and 'town' are what tells
# Man City from Man United. Instead the abbreviations are expanded so the two
# spellings meet in the middle.
_SUBS = (('utd', 'united'), ('nottm', 'nottingham'), ('wolverhampton', 'wolves'))

# Kickoff times agree across sources in practice, but a fixture can be moved
# between one source's snapshot and another's; allow a modest window so a
# re-scheduled match still matches on name, and rely on the name score to keep
# it honest.
KICKOFF_TOLERANCE = datetime.timedelta(hours=6)

# Both sides of a fixture must look alike, and the best candidate must beat the
# runner-up clearly, or we skip rather than risk submitting to the wrong match.
# The per-side floor is what stops Man United matching a Man City fixture: real
# cross-vocabulary matches score >= 0.95 a side, while that near-miss scores
# 0.67, so a sum-only threshold would let it through on the back of a perfect
# match on the other team.
_MIN_SIDE_SCORE = 0.80
_MIN_NAME_SCORE = 1.70      # out of 2.0 (home + away similarity)
_MIN_MARGIN = 0.15


class ScorersError(Exception):
    pass


def _canon(name):
    """Reduce a team name to a comparable core: lowercase, no punctuation, and
    with the club-name filler that differs between sources stripped out.
    'Nott'm Forest', 'Nottingham Forest FC' and 'Nottm Forest' all collapse
    toward the same stem."""
    # Apostrophes are deleted, not spaced, so "Nott'm" stays one token and can
    # be expanded to "nottingham" rather than splitting into "nott" + "m".
    s = _APOSTROPHE.sub('', (name or '').lower())
    s = _NON_ALNUM.sub(' ', s)
    s = _NOISE.sub(' ', s)
    words = [dict(_SUBS).get(w, w) for w in s.split()]
    return ' '.join(words)


def _similarity(a, b):
    ca, cb = _canon(a), _canon(b)
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0
    # A short canonical stem that is contained in the other ('man' in 'man
    # city') is a strong signal that SequenceMatcher alone under-rates.
    if len(ca) >= 3 and len(cb) >= 3 and (ca in cb or cb in ca):
        return 0.95
    return difflib.SequenceMatcher(None, ca, cb).ratio()


def _parse_time(value):
    if not value:
        return None
    if isinstance(value, datetime.datetime):
        dt = value
    else:
        try:
            dt = datetime.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def match_fixture(fixtures, home, away, kickoff=None):
    """Find the Scorers fixture corresponding to a (home, away, kickoff) pick.

    Returns the fixture dict, or None when no candidate is confidently the best
    one. Skipping is deliberate: a wrong match would submit a prediction against
    somebody else's game, which is far worse than submitting nothing."""
    kickoff = _parse_time(kickoff)
    scored = []
    for f in fixtures:
        ft = _parse_time(f.get('kickoffTime'))
        if kickoff and ft and abs(ft - kickoff) > KICKOFF_TOLERANCE:
            continue
        sh = _similarity(f.get('homeTeam'), home)
        sa = _similarity(f.get('awayTeam'), away)
        if sh < _MIN_SIDE_SCORE or sa < _MIN_SIDE_SCORE:
            continue
        scored.append((sh + sa, f))
    if not scored:
        return None
    scored.sort(key=lambda s: s[0], reverse=True)
    best_score, best = scored[0]
    if best_score < _MIN_NAME_SCORE:
        return None
    if len(scored) > 1 and best_score - scored[1][0] < _MIN_MARGIN:
        return None
    return best


class ScorersClient:
    """Thin wrapper over the Scorers bot API for a single account/API key."""

    def __init__(self, api_key, base_url=None, timeout=30, league_id=None):
        if not api_key:
            raise ScorersError('No Scorers API key configured for this stream')
        self.api_key = api_key
        self.base_url = (base_url or os.environ.get('SCORERS_BASE_URL', DEFAULT_BASE_URL)).rstrip('/')
        self.timeout = timeout
        self.league_id = league_id or os.environ.get('SCORERS_LEAGUE_ID') or None

    def _headers(self):
        return {'Authorization': f'Bearer {self.api_key}', 'Accept': 'application/json'}

    def _params(self, **extra):
        params = dict(extra)
        if self.league_id:
            params['leagueId'] = self.league_id
        return params

    def open_fixtures(self):
        """Fixtures still open for prediction, for this account's league."""
        r = requests.get(f'{self.base_url}/api/v1/fixtures', headers=self._headers(),
                         params=self._params(status='open'), timeout=self.timeout)
        if r.status_code == 401:
            raise ScorersError('Scorers rejected the API key (401)')
        if r.status_code != 200:
            raise ScorersError(f'GET /fixtures HTTP {r.status_code}: {r.text[:200]}')
        return r.json()

    def submit(self, fixture_id, home_score, away_score, is_banker=None):
        """Create or overwrite a prediction. Omitting is_banker leaves the
        account's banker untouched, which is what we want on the scores pass."""
        body = {'fixtureId': fixture_id, 'homeScore': int(home_score), 'awayScore': int(away_score)}
        if is_banker is not None:
            body['isBanker'] = bool(is_banker)
        r = requests.post(f'{self.base_url}/api/v1/predictions', headers=self._headers(),
                          params=self._params(), json=body, timeout=self.timeout)
        if r.status_code == 401:
            raise ScorersError('Scorers rejected the API key (401)')
        if r.status_code != 200:
            # 422 is an expected, per-fixture outcome (kicked off, banker
            # locked) rather than a failure of the run, so hand it back to the
            # caller to record instead of aborting the whole submission.
            detail = ''
            try:
                detail = r.json().get('error', '')
            except ValueError:
                detail = r.text[:200]
            return {'ok': False, 'status': r.status_code, 'error': detail}
        return {'ok': True, 'status': 200, 'prediction': r.json().get('prediction', {})}

    def leaderboard(self):
        r = requests.get(f'{self.base_url}/api/v1/leaderboard', headers=self._headers(),
                         params=self._params(), timeout=self.timeout)
        if r.status_code != 200:
            raise ScorersError(f'GET /leaderboard HTTP {r.status_code}: {r.text[:200]}')
        return r.json()
