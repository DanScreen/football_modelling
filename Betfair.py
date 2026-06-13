#!/usr/bin/env python
# coding: utf-8
"""Minimal Betfair Exchange API client.

Login (certificate or interactive) + read-only access to the CORRECT_SCORE
market, so we can source real per-scoreline odds (rather than inferring them
from 1X2). Uses the free Delayed Application Key — prices are delayed but fine
for daily modelling.

Accounts with two-factor auth enabled (required for app keys) cannot use the
interactive login endpoint, so certificate login is the default when a client
cert is available.
"""

import os
import re
import datetime
import tempfile

import requests

IDENTITY_URL = 'https://identitysso.betfair.com/api/login'
CERT_LOGIN_URL = 'https://identitysso-cert.betfair.com/api/certlogin'
BETTING_URL = 'https://api.betfair.com/exchange/betting/json-rpc/v1'
SOCCER_EVENT_TYPE_ID = '1'

# Betfair caps listMarketBook at 25 markets per call when pulling best offers.
_MARKET_BOOK_BATCH = 25

_SCORE_RE = re.compile(r'^\s*(\d+)\s*-\s*(\d+)\s*$')


class BetfairError(Exception):
    pass


def _resolve_cert():
    """Return a (cert_path, key_path) tuple for certificate login, or None.

    Supports either direct file paths (BETFAIR_CERT_FILE / BETFAIR_KEY_FILE) or
    raw PEM contents (BETFAIR_CERT / BETFAIR_KEY) which are written to temp
    files — the latter suits Cloud Run, where the cert is injected as a secret.
    """
    cert_file = os.environ.get('BETFAIR_CERT_FILE')
    key_file = os.environ.get('BETFAIR_KEY_FILE')
    if cert_file and key_file:
        return cert_file, key_file

    cert_pem = os.environ.get('BETFAIR_CERT')
    key_pem = os.environ.get('BETFAIR_KEY')
    if cert_pem and key_pem:
        cf = tempfile.NamedTemporaryFile('w', suffix='.crt', delete=False)
        cf.write(cert_pem)
        cf.close()
        kf = tempfile.NamedTemporaryFile('w', suffix='.key', delete=False)
        kf.write(key_pem)
        kf.close()
        return cf.name, kf.name

    return None


def login(app_key, username, password, cert=None):
    """Return a session token (used as X-Authentication).

    If a client certificate is supplied, use non-interactive certificate login
    (the only path that works for 2FA-enabled accounts); otherwise fall back to
    interactive username/password login.
    """
    if cert:
        resp = requests.post(
            CERT_LOGIN_URL,
            data={'username': username, 'password': password},
            headers={
                'X-Application': app_key,
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json',
            },
            cert=cert,
            timeout=30,
        )
        if resp.status_code != 200:
            raise BetfairError(f'cert login HTTP {resp.status_code}: {resp.text}')
        body = resp.json()
        if body.get('loginStatus') != 'SUCCESS':
            raise BetfairError(f"cert login failed: {body.get('loginStatus')}")
        return body['sessionToken']

    resp = requests.post(
        IDENTITY_URL,
        data={'username': username, 'password': password},
        headers={
            'X-Application': app_key,
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise BetfairError(f'login HTTP {resp.status_code}: {resp.text}')
    body = resp.json()
    if body.get('status') != 'SUCCESS':
        raise BetfairError(f"login failed: {body.get('status')} / {body.get('error')}")
    return body['token']


def _rpc(method, params, app_key, token):
    resp = requests.post(
        BETTING_URL,
        json={'jsonrpc': '2.0', 'method': f'SportsAPING/v1.0/{method}', 'params': params, 'id': 1},
        headers={
            'X-Application': app_key,
            'X-Authentication': token,
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise BetfairError(f'{method} HTTP {resp.status_code}: {resp.text}')
    body = resp.json()
    if 'error' in body:
        raise BetfairError(f'{method} error: {body["error"]}')
    return body['result']


def _parse_time(iso_str):
    if not iso_str:
        return None
    return datetime.datetime.fromisoformat(iso_str.replace('Z', '+00:00'))


def _classify_runner(name):
    """Return ('exact', (home, away)) for a scoreline runner like '2 - 1',
    or ('home'|'draw'|'away', None) for an 'Any Other ...' bucket."""
    m = _SCORE_RE.match(name)
    if m:
        return 'exact', (int(m.group(1)), int(m.group(2)))
    low = name.lower()
    if 'home' in low:
        return 'home', None
    if 'away' in low:
        return 'away', None
    if 'draw' in low or 'unquoted' in low:
        return 'draw', None
    return 'unknown', None


def _devig(raw_probs):
    total = sum(raw_probs.values())
    if total <= 0:
        return {}, 0.0
    return {k: v / total for k, v in raw_probs.items()}, total


def get_worldcup_correct_score(app_key=None, username=None, password=None,
                               query=None, max_markets=20):
    """Fetch upcoming FIFA World Cup CORRECT_SCORE markets and return, per match,
    the de-vigged probability of each explicit scoreline plus the 'any other
    home/draw/away win' buckets.

    Returns a list of dicts:
        {
          'match': [home, away],
          'time': datetime,
          'scorelines': {(home, away): prob, ...},   # explicit, de-vigged
          'other': {'home': p, 'draw': p, 'away': p}, # de-vigged buckets
          'overround': float,                          # pre-de-vig book sum
        }
    """
    app_key = app_key or os.environ.get('BETFAIR_APP_KEY')
    username = username or os.environ.get('BETFAIR_USERNAME')
    password = password or os.environ.get('BETFAIR_PASSWORD')
    query = query or os.environ.get('BETFAIR_WC_QUERY', 'FIFA World Cup')
    if not (app_key and username and password):
        raise BetfairError('BETFAIR_APP_KEY, BETFAIR_USERNAME and BETFAIR_PASSWORD must be set')

    token = login(app_key, username, password, cert=_resolve_cert())
    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    catalogue = _rpc('listMarketCatalogue', {
        'filter': {
            'eventTypeIds': [SOCCER_EVENT_TYPE_ID],
            'textQuery': query,
            'marketTypeCodes': ['CORRECT_SCORE'],
            'marketStartTime': {'from': now_iso},
        },
        'marketProjection': ['EVENT', 'RUNNER_DESCRIPTION', 'MARKET_START_TIME'],
        'sort': 'FIRST_TO_START',
        'maxResults': max_markets,
    }, app_key, token)

    if not catalogue:
        return []

    # selectionId -> runner name, per market, plus event/time metadata.
    meta = {}
    for mkt in catalogue:
        market_id = mkt['marketId']
        event = mkt.get('event', {})
        event_name = event.get('name', '')
        if ' v ' in event_name:
            home, away = event_name.split(' v ', 1)
        elif ' vs ' in event_name.lower():
            parts = re.split(r'\s+vs\s+', event_name, maxsplit=1, flags=re.IGNORECASE)
            home, away = parts[0], parts[1]
        else:
            home, away = event_name, ''
        meta[market_id] = {
            'home': home.strip(),
            'away': away.strip(),
            'time': _parse_time(event.get('openDate') or mkt.get('marketStartTime')),
            'runners': {r['selectionId']: r['runnerName'] for r in mkt.get('runners', [])},
        }

    # Pull best-back prices, batching to respect Betfair's per-call market cap.
    market_ids = list(meta.keys())
    books = {}
    for i in range(0, len(market_ids), _MARKET_BOOK_BATCH):
        batch = market_ids[i:i + _MARKET_BOOK_BATCH]
        result = _rpc('listMarketBook', {
            'marketIds': batch,
            'priceProjection': {'priceData': ['EX_BEST_OFFERS']},
        }, app_key, token)
        for book in result:
            books[book['marketId']] = book

    matches = []
    for market_id, info in meta.items():
        book = books.get(market_id)
        if not book:
            continue
        raw_scores = {}
        raw_other = {'home': 0.0, 'draw': 0.0, 'away': 0.0}
        for runner in book.get('runners', []):
            offers = runner.get('ex', {}).get('availableToBack', [])
            if not offers:
                continue
            price = offers[0]['price']  # best back, returned best-first
            if not price or price <= 1.0:
                continue
            implied = 1.0 / price
            name = info['runners'].get(runner['selectionId'], '')
            kind, score = _classify_runner(name)
            if kind == 'exact':
                raw_scores[score] = raw_scores.get(score, 0.0) + implied
            elif kind in raw_other:
                raw_other[kind] += implied

        combined = dict(raw_scores)
        for region, p in raw_other.items():
            if p > 0:
                combined[('other', region)] = p
        devigged, overround = _devig(combined)

        scorelines = {k: v for k, v in devigged.items() if isinstance(k, tuple) and isinstance(k[0], int)}
        other = {'home': 0.0, 'draw': 0.0, 'away': 0.0}
        for k, v in devigged.items():
            if isinstance(k, tuple) and k[0] == 'other':
                other[k[1]] = v

        matches.append({
            'match': [info['home'], info['away']],
            'time': info['time'],
            'scorelines': scorelines,
            'other': other,
            'overround': overround,
        })

    matches.sort(key=lambda m: (m['time'] or datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)))
    return matches
