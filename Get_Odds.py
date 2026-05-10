#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import json
import requests
import datetime

basetime = datetime.datetime(year=1970, month=1, day=1, hour=1)

def get_odds(sport_key='soccer_epl'):
    api_key = os.environ.get('ODDS_API_KEY', '8eed4238adf2be8a8f66ee3c2624c5ee')
    if not api_key:
        raise ValueError("ODDS_API_KEY environment variable not set")

    odds_response = requests.get(
        'https://api.the-odds-api.com/v4/sports/{}/odds'.format(sport_key),
        params={
            'api_key': api_key,
            'regions': 'uk',
            'markets': 'h2h',
            'oddsFormat': 'decimal',
        },
        timeout=30,
    )

    if odds_response.status_code != 200:
        print('Error fetching odds: {} {}'.format(odds_response.status_code, odds_response.text))
        return

    print('Remaining requests:', odds_response.headers.get('x-requests-remaining'))
    print('Used requests:', odds_response.headers.get('x-requests-used'))

    odds_json = odds_response.json()

    betting_odds = []
    for match in odds_json:
        home_team = match['home_team']
        away_team = match['away_team']
        commence_time = datetime.datetime.fromisoformat(match['commence_time'].replace('Z', '+00:00'))

        best_odds = {
            'match': [home_team, away_team],
            'time': commence_time,
            'event': ['Home', 'Draw', 'Away'],
        }
        betting_odds.append(best_odds)

    return betting_odds
