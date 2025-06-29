#!/usr/bin/env python
# coding: utf-8

# In[4]:


## script to get best odds from The Odds API
import numpy as np
import json
import requests
import datetime
basetime = datetime.datetime(year=1970, month=1, day=1, hour=1)
def get_odds(sport_key = 'soccer_epl'):
    # An api key is emailed to you when you sign up to a plan
    api_key = '8eed4238adf2be8a8f66ee3c2624c5ee'

    # First get a list of in-season sports
    sports_response = requests.get('https://api.the-odds-api.com/v3/sports', params={
        'api_key': api_key})

    sports_json = json.loads(sports_response.text)
    if not sports_json['success']:
        print('There was a problem with the sports request:',
            sports_json['msg'])
        return

    odds_response = requests.get('https://api.the-odds-api.com/v3/odds', params={
        'api_key': api_key,
        'sport': sport_key,
        'region': 'uk', # uk | us | eu | au
        'mkt': 'h2h' # h2h | spreads | totals
    })

    odds_json = json.loads(odds_response.text)
    if not odds_json['success']:
        print('There was a problem with the odds request:',
            odds_json['msg'])
        return

    else:
        print('Remaining requests', odds_response.headers['x-requests-remaining'])
        print('Used requests', odds_response.headers['x-requests-used'])
        print(odds_json['data'])
    
    betting_odds = []
    for i in range(len(odds_json['data'])):
        match = odds_json['data'][i]

        home_index = 0
        away_index = 1
        if match['teams'][0] != match['home_team']:
            match['teams'].reverse()
            home_index = 1
            away_index = 0


#         sites = []
#         Ohome = []
#         Odraw = []
#         Oaway = []
#         for site in match['sites']:
#             if site['site_nice'] in ['Marathon Bet', '1xBet', 'Matchbook']:
#                 continue
#             sites.append(site['site_nice'])
#             Ohome.append(site['odds']['h2h'][home_index])
#             Odraw.append(site['odds']['h2h'][2])
#             Oaway.append(site['odds']['h2h'][away_index])
        
#         if not (len(sites)==len(Ohome)==len(Odraw)==len(Oaway)):
#             print('Varying lengths of scraped data')
        
#         print(Ohome)
#         print(Odraw)
#         index_home = np.argmax(Ohome)
#         index_draw = np.argmax(Odraw)
#         index_away = np.argmax(Oaway)
        best_odds = {'match': match['teams'],
                     'time':basetime + datetime.timedelta(seconds=match['commence_time']),
                     'event': ['Home', 'Draw', 'Away'], 
#                      'Bookie': [sites[index_home], sites[index_draw], sites[index_away]],
#                      'Odds': [Ohome[index_home], Odraw[index_draw], Oaway[index_away]]
                    }
#        best_odds['Probs'] = list(1/np.array(best_odds['Odds']))
        betting_odds.append(best_odds)
        
    return betting_odds

