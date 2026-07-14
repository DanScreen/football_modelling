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
import difflib
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


def _fetch_to_qualify(query, now_iso, app_key, token, max_markets):
    """Fetch TO_QUALIFY markets (only present for knockout ties) and return
    {event_id: {runner_name: devigged_prob}}. The presence of an entry for an
    event is itself the signal that the tie is a knockout (extra time applies)."""
    catalogue = _rpc('listMarketCatalogue', {
        'filter': {
            'eventTypeIds': [SOCCER_EVENT_TYPE_ID],
            'textQuery': query,
            'marketTypeCodes': ['TO_QUALIFY'],
            'marketStartTime': {'from': now_iso},
        },
        'marketProjection': ['EVENT', 'RUNNER_DESCRIPTION'],
        # Same sort as the correct-score query so both cap the same earliest
        # events when max_markets bites — otherwise event sets can diverge.
        'sort': 'FIRST_TO_START',
        'maxResults': max_markets,
    }, app_key, token)
    if not catalogue:
        return {}

    meta = {}
    for mkt in catalogue:
        event_id = mkt.get('event', {}).get('id')
        if not event_id:
            continue
        meta[mkt['marketId']] = {
            'event_id': event_id,
            'runners': {r['selectionId']: r['runnerName'] for r in mkt.get('runners', [])},
        }

    market_ids = list(meta.keys())
    by_event = {}
    for i in range(0, len(market_ids), _MARKET_BOOK_BATCH):
        batch = market_ids[i:i + _MARKET_BOOK_BATCH]
        result = _rpc('listMarketBook', {
            'marketIds': batch,
            'priceProjection': {'priceData': ['EX_BEST_OFFERS']},
        }, app_key, token)
        for book in result:
            info = meta.get(book['marketId'])
            if not info:
                continue
            raw = {}
            for runner in book.get('runners', []):
                offers = runner.get('ex', {}).get('availableToBack', [])
                if not offers:
                    continue
                price = offers[0]['price']
                if not price or price <= 1.0:
                    continue
                name = info['runners'].get(runner['selectionId'], '')
                if name:
                    raw[name] = raw.get(name, 0.0) + 1.0 / price
            devigged, _ = _devig(raw)
            # If an event has more than one TO_QUALIFY market, keep the one with
            # the most priced runners rather than letting an arbitrary last win.
            existing = by_event.get(info['event_id'])
            if devigged and (existing is None or len(devigged) > len(existing)):
                by_event[info['event_id']] = devigged
    return by_event


def _match_qualify(qualify_runners, home, away):
    """Map a {runner_name: prob} dict from the To Qualify market onto the
    home/away teams, tolerating minor name differences between markets."""
    if not qualify_runners:
        return None
    names = list(qualify_runners)
    if home in qualify_runners and away in qualify_runners:
        return {'home': qualify_runners[home], 'away': qualify_runners[away]}
    if len(names) != 2:
        return None
    # Pick the runner->team orientation with the best combined name similarity.
    # Refuse an ambiguous or weak match (return None -> the tie falls back to the
    # safe 90-minute basis) rather than risk swapping home/away qualify probs.
    a, b = names

    def sim(x, y):
        return difflib.SequenceMatcher(None, x.lower(), y.lower()).ratio()

    ab = sim(a, home) + sim(b, away)  # a=home, b=away
    ba = sim(b, home) + sim(a, away)  # b=home, a=away
    if max(ab, ba) < 1.0 or abs(ab - ba) < 0.2:
        return None
    if ab >= ba:
        return {'home': qualify_runners[a], 'away': qualify_runners[b]}
    return {'home': qualify_runners[b], 'away': qualify_runners[a]}


def get_worldcup_correct_score(app_key=None, username=None, password=None,
                               query=None, max_markets=20, include_qualify=True):
    """Fetch upcoming FIFA World Cup CORRECT_SCORE markets and return, per match,
    the de-vigged probability of each explicit scoreline plus the 'any other
    home/draw/away win' buckets.

    CORRECT_SCORE settles on 90 minutes only. When include_qualify is set we also
    pull the TO_QUALIFY market (present only for knockout ties) in the same
    session; its presence flags the tie as a knockout, and its de-vigged prices
    give each side's probability of advancing (after extra time and penalties),
    used downstream to build the 120-minute score distribution SuperBru scores on.

    Returns a list of dicts:
        {
          'match': [home, away],
          'event_id': str,
          'time': datetime,
          'scorelines': {(home, away): prob, ...},   # explicit, de-vigged, 90 min
          'other': {'home': p, 'draw': p, 'away': p}, # de-vigged buckets, 90 min
          'overround': float,                          # pre-de-vig book sum
          'knockout': bool,                            # TO_QUALIFY market exists
          'qualify': {'home': p, 'away': p} | None,    # de-vigged, incl. ET + pens
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
            'event_id': event.get('id'),
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
            'event_id': info.get('event_id'),
            'time': info['time'],
            'scorelines': scorelines,
            'other': other,
            'overround': overround,
            'knockout': False,
            'qualify': None,
        })

    if include_qualify:
        # Qualify data is an optional enrichment: a failure here (including a
        # network/timeout error, which is not a BetfairError) must never take
        # down the core correct-score response, so degrade to no knockout data.
        try:
            qualify_by_event = _fetch_to_qualify(query, now_iso, app_key, token, max_markets)
        except Exception as e:
            print(f'To Qualify fetch failed, continuing without knockout data: {e}')
            qualify_by_event = {}
        for m in matches:
            runners = qualify_by_event.get(m['event_id'])
            if not runners:
                continue
            mapped = _match_qualify(runners, m['match'][0], m['match'][1])
            if mapped:
                m['knockout'] = True
                m['qualify'] = mapped

    matches.sort(key=lambda m: (m['time'] or datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)))
    return matches
