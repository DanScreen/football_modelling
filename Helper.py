#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import re
from scipy.stats import poisson, skellam
import datetime


# In[2]:


def BS(X,P):
    if len(X) != len(P):
        print('Arrays of different length given as input')
        return
        
    X = np.array(X)
    P = np.array(P)
    output = (X-P)**2
    return output.sum(axis=1)

def LS(X,P):
    if len(X) != len(P):
        print('Arrays of different length given as input')
        return
        
    X = np.array(X)
    P = np.array(P)
    summa = 0
    n = X.shape[0]
    for i in range(n):
        summa = summa + sum(X[i,]*np.log(P[i,])) + sum((1-X[i,])*np.log(1-P[i,]))
    return -1/n*summa

def FTRtoZ(FTR):
    converter = {'H':[1,0,0], 'D':[0,1,0], 'A':[0,0,1]}
    Z = []
    for i in range(len(FTR)):
        x = FTR[i]
        z = converter[x]
        Z.append(z)
    return np.array(Z)

def logit(x):
    return 1/(1+np.exp(-x))

def inv_logit(x):
    return np.log(x/(1-x))

def season_file(season, league_str):
    return 'AutoData/'+str(season)+str(league_str)+'.csv'

def available_seasons(SEA, league_str):
    """Drop seasons whose main league CSV hasn't been downloaded yet.

    football-data.co.uk only publishes a season's file once it kicks off, so a
    range that runs to the current season will reference a file that may not
    exist. Seasons are skipped rather than raising, so training always uses
    whatever data is on disk."""
    present = [i for i in SEA if os.path.exists(season_file(i, league_str))]
    missing = [i for i in SEA if i not in present]
    if missing:
        print('Skipping seasons with no '+str(league_str)+' data: '+
              ', '.join(str(i) for i in missing))
    if not present:
        raise FileNotFoundError(
            'No '+str(league_str)+' season files found in AutoData/ for seasons '+
            str(list(SEA))+'. Download the data first.')
    return present

def read_season(season, league_str):
    """Read a season's CSV, or return an empty frame if it isn't downloaded.

    Used for the feeder/parent divisions, which are only needed to work out who
    was promoted/relegated into the *following* season — so a missing current
    season file costs nothing until that next season exists."""
    path = season_file(season, league_str)
    if not os.path.exists(path):
        print('No data for season '+str(season)+' '+str(league_str)+' - continuing without it')
        return pd.DataFrame(columns=['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'])
    return read_football_data(path)

def season_teams(data):
    """All teams in a season, from home *and* away fixtures.

    Part-played seasons matter here: after one round only half the division has
    played at home, so keying off HomeTeam alone would miss the rest. For a
    completed season this is identical to the home-team list."""
    if data.empty:
        return np.array([], dtype=object)
    return np.unique(np.concatenate([data['HomeTeam'].values, data['AwayTeam'].values]))


# In[3]:


class league_fast():
    def __init__(self, teams=None, p_alpha=None, q_alpha=None, alpha_hat=None, p_beta=None, q_beta=None, beta_hat=None,
                p_gamma=None, q_gamma=None, gamma_hat=None, w=0.9879 , w_b=0.77767, w3=0.9984992, delta=10, 
                 delta_g=713.4048222, 
                 promoted=dict({'p_alpha':44.28, 'q_alpha':53.71, 'p_beta':43.43, 'q_beta':39.29, 
                           'p_gamma':1.45*713.4048222, 'q_gamma':713.4048222}), 
                 relegated=dict({'p_alpha':53.71, 'q_alpha':44.28, 'p_beta':39.29, 'q_beta':43.43, 
                           'p_gamma':1.45*713.4048222, 'q_gamma':713.4048222}), trained_data=None):
        self.teams = teams
        if self.teams:
            self.NT = len(teams)
        self.p_alpha = p_alpha
        self.q_alpha = q_alpha
        self.alpha_hat = alpha_hat
        self.p_beta = p_beta
        self.q_beta = q_beta
        self.beta_hat = beta_hat
        self.p_gamma = p_gamma
        self.q_gamma = q_gamma
        self.gamma_hat = gamma_hat
        self.w = w
        self.w3 = w3
        self.w_b = w_b
        self.delta = delta
        self.delta_g = delta_g
        self.promoted = promoted
        self.relegated = relegated
        self.trained_data = trained_data
        self.seasons_trained = []
        self.rolled_forward_to = None
    
    def initialise(self, teams):
        self.teams = teams
        self.NT = len(teams)
        self.p_alpha = np.array([self.delta]*self.NT, dtype=float)
        self.q_alpha = np.array([self.delta]*self.NT, dtype=float)
        self.alpha_hat = (self.p_alpha)/self.q_alpha
        
        self.p_beta = np.array([self.delta]*self.NT, dtype=float)
        self.q_beta = np.array([self.delta]*self.NT, dtype=float)
        self.beta_hat = self.p_beta/self.q_beta
        
        self.p_gamma = np.array([1.45*self.delta_g]*self.NT, dtype=float)
        self.q_gamma = np.array([self.delta_g]*self.NT, dtype=float)
        self.gamma_hat = (self.p_gamma-1)/self.q_gamma
    
    def train(self, data):
        data['FTHG'] = pd.to_numeric(data['FTHG'])
        data['FTAG'] = pd.to_numeric(data['FTAG'])
        data.loc[data['FTHG'] > 5, 'FTHG'] = 5
        data.loc[data['FTAG'] > 5, 'FTAG'] = 5
        # iterate through data
        for i in range(data.shape[0]):
            match = data.iloc[[i]]
            # get indices of home and away sides
            HT = int(np.arange(self.NT)[self.teams==match.iloc[0].loc['HomeTeam']][0])
            AT = int(np.arange(self.NT)[self.teams==match.iloc[0].loc['AwayTeam']][0])
            X = int(match['FTHG'].iloc[0])
            Y = int(match['FTAG'].iloc[0])
            
            self.p_alpha[HT] = self.w*self.p_alpha[HT]+X
            self.q_alpha[HT] = self.w*self.q_alpha[HT]+self.beta_hat[AT]*self.gamma_hat[HT]
            self.alpha_hat[HT] = (self.p_alpha[HT]-1)/self.q_alpha[HT]

            self.p_alpha[AT] = self.w*self.p_alpha[AT]+Y
            self.q_alpha[AT] = self.w*self.q_alpha[AT]+self.beta_hat[HT]
            self.alpha_hat[AT] = (self.p_alpha[AT]-1)/self.q_alpha[AT]

            self.p_beta[HT] = self.w*self.p_beta[HT]+Y
            self.q_beta[HT] = self.w*self.q_beta[HT]+self.alpha_hat[AT]
            self.beta_hat[HT] = (self.p_beta[HT]-1)/self.q_beta[HT]

            self.p_beta[AT] = self.w*self.p_beta[AT]+X
            self.q_beta[AT] = self.w*self.q_beta[AT]+self.alpha_hat[HT]*self.gamma_hat[HT]
            self.beta_hat[AT] = (self.p_beta[AT]-1)/self.q_beta[AT]

            self.p_gamma[HT] = self.w3*self.p_gamma[HT]+X
            self.q_gamma[HT] = self.w3*self.q_gamma[HT]+self.alpha_hat[HT]*self.beta_hat[AT]
            self.gamma_hat[HT] = (self.p_gamma[HT]-1)/self.q_gamma[HT]

    def predict(self, HomeTeam, AwayTeam):
        HT = int(np.arange(self.NT)[self.teams==HomeTeam][0])
        AT = int(np.arange(self.NT)[self.teams==AwayTeam][0])
        LambdaH = self.alpha_hat[HT]*self.beta_hat[AT]*self.gamma_hat[HT]
        LambdaA = self.alpha_hat[AT]*self.beta_hat[HT]
        
        home_goals = np.zeros(9)
        away_goals = np.zeros(9)
        for i in range(8):
            home_goals[i] = poisson.pmf(i, LambdaH)
            away_goals[i] = poisson.pmf(i, LambdaA)
        home_goals[8] = 1-sum(home_goals)
        away_goals[8] = 1-sum(away_goals)
        scores = np.zeros((9,9))
        for i in range(9):
            for j in range(9):
                scores[i,j] = home_goals[i]*away_goals[j]
        
        phome = np.tril(scores, -1).sum()
        pdraw = sum(np.diag(scores))
        paway = np.triu(scores, 1).sum()
        
        # most likely result
        result = np.where(scores==np.max(scores))
        result = list(result)
        ml_result = [int(result[0]), int(result[1])]
        
        return({'matrix':scores, 'outcomes':[phome, pdraw, paway], 'result':ml_result})
    
    def new_season(self, teams_out, teams_promoted_in, teams_relegated_in=None):
        # record variables belonging to each team
        tracker=dict({'teams':self.teams, 'p_alpha':self.p_alpha, 'q_alpha':self.q_alpha, 'p_beta':self.p_beta, 
                      'q_beta':self.q_beta, 'p_gamma':self.p_gamma, 'q_gamma':self.q_gamma})
        teams_df = pd.DataFrame(tracker)
        # remove teams exiting league
        teams_out_index = []
        for i in range(len(self.teams)):
            if self.teams[i] in teams_out:
                teams_out_index.append(i)
        self.teams = np.delete(self.teams, teams_out_index)
        # add new teams to the league
        self.teams = np.append(self.teams, teams_promoted_in)
        if teams_relegated_in:
            self.teams = np.append(self.teams, teams_relegated_in)
        self.teams = np.array(sorted(self.teams))

        self.p_alpha = np.array([])
        self.q_alpha = np.array([])
        self.p_beta = np.array([])
        self.q_beta = np.array([])
        self.p_gamma = np.array([])
        self.q_gamma = np.array([])
        for i in range(self.NT):
            if self.teams[i] in list(teams_df['teams']):
                team_data = teams_df[teams_df['teams']==self.teams[i]]
                w_b = self.w_b
                w3 = self.w3
            elif self.teams[i] in list(teams_promoted_in):
                team_data = self.promoted
                w_b = 1
                w3 = 1
            elif self.teams[i] in list(teams_relegated_in):
                team_data = self.relegated
                w_b = 1
                w3 = 1
                
            self.p_alpha = np.append(self.p_alpha, w_b*float(team_data['p_alpha'].iloc[0] if hasattr(team_data['p_alpha'], 'iloc') else team_data['p_alpha']))
            self.q_alpha = np.append(self.q_alpha, w_b*float(team_data['q_alpha'].iloc[0] if hasattr(team_data['q_alpha'], 'iloc') else team_data['q_alpha']))
            self.p_beta = np.append(self.p_beta, w_b*float(team_data['p_beta'].iloc[0] if hasattr(team_data['p_beta'], 'iloc') else team_data['p_beta']))
            self.q_beta = np.append(self.q_beta, w_b*float(team_data['q_beta'].iloc[0] if hasattr(team_data['q_beta'], 'iloc') else team_data['q_beta']))
            self.p_gamma = np.append(self.p_gamma, w3*float(team_data['p_gamma'].iloc[0] if hasattr(team_data['p_gamma'], 'iloc') else team_data['p_gamma']))
            self.q_gamma = np.append(self.q_gamma, w3*float(team_data['q_gamma'].iloc[0] if hasattr(team_data['q_gamma'], 'iloc') else team_data['q_gamma']))
            
        self.alpha_hat = (self.p_alpha-1)/self.q_alpha
        self.beta_hat = (self.p_beta-1)/self.q_beta
        self.gamma_hat = (self.p_gamma-1)/self.q_gamma
        
    def start_next_season(self, season, league_below='E1', league_below_2='E2'):
        """Advance the model into `season` before that division's own file exists.

        football-data.co.uk publishes a division's CSV only once it kicks off, and
        the Premier League lags the EFL by a week or two, so there is a window each
        August where the new season has started but there are no matches to train
        on. The new line-up is already implied by the divisions below, though:
        whoever went down turns up in the feeder division, and whoever came up is
        missing from both the feeder division and the one below that. Applying the
        promoted/relegated priors now means the new sides can be predicted straight
        away instead of erroring as unknown teams.

        Returns (teams_out, promoted_in) if the model was advanced, else None.
        """
        below_now = read_season(season, league_below)
        below_before = read_season(season - 1, league_below)
        below_2_now = read_season(season, league_below_2)
        if below_now.empty or below_before.empty or below_2_now.empty:
            print('Cannot roll forward into ' + str(season) +
                  ': need ' + str(league_below) + ' for ' + str(season - 1) + ' and ' +
                  str(season) + ' plus ' + str(league_below_2) + ' for ' + str(season))
            return None

        now_below = set(season_teams(below_now))
        teams_out = sorted(set(self.teams) & now_below)
        promoted_in = sorted(set(season_teams(below_before)) - now_below - set(season_teams(below_2_now)))

        # new_season rebuilds the parameter arrays over a fixed roster size, so a
        # lopsided swap would silently corrupt them. Bail out instead.
        if not teams_out or len(teams_out) != len(promoted_in):
            print('Cannot roll forward into ' + str(season) + ': inferred ' +
                  str(len(teams_out)) + ' out / ' + str(len(promoted_in)) + ' in ' +
                  '(' + str(teams_out) + ' / ' + str(promoted_in) + ')')
            return None

        self.new_season(teams_out, promoted_in)
        self.rolled_forward_to = season
        print('Rolled forward into ' + str(season) + ': out ' + str(teams_out) +
              ', in ' + str(promoted_in))
        return teams_out, promoted_in

    def train_all(self, league_str, league_below=None, league_above=None, SEA = list(range(1996, 2021))):
        SEA = available_seasons(SEA, league_str)
        self.seasons_trained = list(SEA)
        
        data = read_football_data(season_file(SEA[0], league_str))
        teams = season_teams(data)

        self.teams = teams
        self.NT = len(teams)

        if league_below:
            data_below = read_season(SEA[0], league_below)
            teams_below = season_teams(data_below)

        if league_above:
            data_above = read_season(SEA[0], league_above)
            teams_above = season_teams(data_above)

        print('Season: ' + str(SEA[0]), end="\r")
        self.initialise(teams)
        self.train(data)
        promoted_in=None
        relegated_in=None
        for i in range(1, len(SEA)):
            print('Season: ' + str(SEA[i]), end="\r")
            old_data = data
            old_teams = teams
            data = read_football_data(season_file(SEA[i], league_str))
            teams = season_teams(data)
            teams_out = list(set(old_teams) - set(teams))

            if league_below:
                old_data_below = data_below
                old_teams_below = teams_below
                data_below = read_season(SEA[i], league_below)
                teams_below = season_teams(data_below)

            if league_above:
                old_data_above = data_above
                old_teams_above = teams_above
                data_above = read_season(SEA[i], league_above)
                teams_above = season_teams(data_above)

            if league_below:
                promoted_in =  sorted(list(set(old_teams_below) & set(teams)))
            if league_above:
                relegated_in = sorted(list(set(old_teams_above) & set(teams)))

            if not (league_below or league_above):
                promoted_in =  sorted(list(set(teams) - set(old_teams)))

            self.new_season(teams_out, promoted_in, relegated_in)
            self.train(data)
        print('Training Complete')

class league():
    def __init__(self, teams=None, p_alpha=None, q_alpha=None, alpha_hat=None, p_beta=None, q_beta=None, beta_hat=None,
                p_gamma=None, q_gamma=None, gamma_hat=None, w=0.9879 , w_b=0.77767, w3=0.9984992, delta=10, 
                 delta_g=713.4048222, 
                 promoted=dict({'p_alpha':44.28, 'q_alpha':53.71, 'p_beta':43.43, 'q_beta':39.29, 
                           'p_gamma':1.45*713.4048222, 'q_gamma':713.4048222}), 
                 relegated=dict({'p_alpha':53.71, 'q_alpha':44.28, 'p_beta':39.29, 'q_beta':43.43, 
                           'p_gamma':1.45*713.4048222, 'q_gamma':713.4048222}), trained_data=pd.DataFrame()):
        self.teams = teams
        if self.teams:
            self.NT = len(teams)
        self.p_alpha = p_alpha
        self.q_alpha = q_alpha
        self.alpha_hat = alpha_hat
        self.p_beta = p_beta
        self.q_beta = q_beta
        self.beta_hat = beta_hat
        self.p_gamma = p_gamma
        self.q_gamma = q_gamma
        self.gamma_hat = gamma_hat
        self.w = w
        self.w3 = w3
        self.w_b = w_b
        self.delta = delta
        self.delta_g = delta_g
        self.promoted = promoted
        self.relegated = relegated
        self.trained_data = trained_data
        self.seasons_trained = []
        self.rolled_forward_to = None
    
    def initialise(self, teams):
        self.teams = teams
        self.NT = len(teams)
        self.p_alpha = np.array([self.delta]*self.NT, dtype=float)
        self.q_alpha = np.array([self.delta]*self.NT, dtype=float)
        self.alpha_hat = (self.p_alpha)/self.q_alpha
        
        self.p_beta = np.array([self.delta]*self.NT, dtype=float)
        self.q_beta = np.array([self.delta]*self.NT, dtype=float)
        self.beta_hat = self.p_beta/self.q_beta
        
        self.p_gamma = np.array([1.45*self.delta_g]*self.NT, dtype=float)
        self.q_gamma = np.array([self.delta_g]*self.NT, dtype=float)
        self.gamma_hat = (self.p_gamma-1)/self.q_gamma
    
    def train(self, data):
        data['FTHG'] = pd.to_numeric(data['FTHG'])
        data['FTAG'] = pd.to_numeric(data['FTAG'])
        data.loc[data['FTHG'] > 5, 'FTHG'] = 5
        data.loc[data['FTAG'] > 5, 'FTAG'] = 5
        
        data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
        
        lambdasH = []
        lambdasA = []
        # iterate through data
        for i in range(data.shape[0]):
            match = data.iloc[[i]]
            # get indices of home and away sides
            HT = int(np.arange(self.NT)[self.teams==match.iloc[0].loc['HomeTeam']][0])
            AT = int(np.arange(self.NT)[self.teams==match.iloc[0].loc['AwayTeam']][0])
            # get home and away lambda
            lambdaH = self.alpha_hat[HT]*self.beta_hat[AT]*self.gamma_hat[HT]
            lambdasH.append(lambdaH)
            
            lambdaA = self.alpha_hat[AT]*self.beta_hat[HT]
            lambdasA.append(lambdaA)
            
            X = int(match['FTHG'].iloc[0])
            Y = int(match['FTAG'].iloc[0])
            
            self.p_alpha[HT] = self.w*self.p_alpha[HT]+X
            self.q_alpha[HT] = self.w*self.q_alpha[HT]+self.beta_hat[AT]*self.gamma_hat[HT]
            self.alpha_hat[HT] = (self.p_alpha[HT]-1)/self.q_alpha[HT]

            self.p_alpha[AT] = self.w*self.p_alpha[AT]+Y
            self.q_alpha[AT] = self.w*self.q_alpha[AT]+self.beta_hat[HT]
            self.alpha_hat[AT] = (self.p_alpha[AT]-1)/self.q_alpha[AT]

            self.p_beta[HT] = self.w*self.p_beta[HT]+Y
            self.q_beta[HT] = self.w*self.q_beta[HT]+self.alpha_hat[AT]
            self.beta_hat[HT] = (self.p_beta[HT]-1)/self.q_beta[HT]

            self.p_beta[AT] = self.w*self.p_beta[AT]+X
            self.q_beta[AT] = self.w*self.q_beta[AT]+self.alpha_hat[HT]*self.gamma_hat[HT]
            self.beta_hat[AT] = (self.p_beta[AT]-1)/self.q_beta[AT]

            self.p_gamma[HT] = self.w3*self.p_gamma[HT]+X
            self.q_gamma[HT] = self.w3*self.q_gamma[HT]+self.alpha_hat[HT]*self.beta_hat[AT]
            self.gamma_hat[HT] = (self.p_gamma[HT]-1)/self.q_gamma[HT]
        
        phome = 1 - skellam.cdf(0, lambdasH, lambdasA)
        pdraw = skellam.pmf(0, lambdasH, lambdasA)
        paway = 1-phome-pdraw
        P = np.zeros((len(data), 3))
        P[:,0] = phome
        P[:,1] = pdraw
        P[:,2] = paway
        Z = FTRtoZ(data['FTR'])
        data['BS'] = BS(Z, P)
        #data['LS'] = LS(Z, P)
        data['PHome'] = phome
        data['PDraw'] = pdraw
        data['PAway'] = paway
        
        self.trained_data = pd.concat([self.trained_data, data], ignore_index=True)

    def predict(self, HomeTeam, AwayTeam):
        HT = int(np.arange(self.NT)[self.teams==HomeTeam][0])
        AT = int(np.arange(self.NT)[self.teams==AwayTeam][0])
        LambdaH = self.alpha_hat[HT]*self.beta_hat[AT]*self.gamma_hat[HT]
        LambdaA = self.alpha_hat[AT]*self.beta_hat[HT]
        
        home_goals = np.zeros(9)
        away_goals = np.zeros(9)
        for i in range(8):
            home_goals[i] = poisson.pmf(i, LambdaH)
            away_goals[i] = poisson.pmf(i, LambdaA)
        home_goals[8] = 1-sum(home_goals)
        away_goals[8] = 1-sum(away_goals)
        scores = np.zeros((9,9))
        for i in range(9):
            for j in range(9):
                scores[i,j] = home_goals[i]*away_goals[j]
        
        phome = np.tril(scores, -1).sum()
        pdraw = sum(np.diag(scores))
        paway = np.triu(scores, 1).sum()
        
        # most likely result
        result = np.where(scores==np.max(scores))
        result = list(result)
        ml_result = [int(result[0]), int(result[1])]
        
        return({'matrix':scores, 'outcomes':[phome, pdraw, paway], 'result':ml_result})
    
    def new_season(self, teams_out, teams_promoted_in, teams_relegated_in=None):
        # record variables belonging to each team
        tracker=dict({'teams':self.teams, 'p_alpha':self.p_alpha, 'q_alpha':self.q_alpha, 'p_beta':self.p_beta, 
                      'q_beta':self.q_beta, 'p_gamma':self.p_gamma, 'q_gamma':self.q_gamma})
        teams_df = pd.DataFrame(tracker)
        # remove teams exiting league
        teams_out_index = []
        for i in range(len(self.teams)):
            if self.teams[i] in teams_out:
                teams_out_index.append(i)
        self.teams = np.delete(self.teams, teams_out_index)
        # add new teams to the league
        self.teams = np.append(self.teams, teams_promoted_in)
        if teams_relegated_in:
            self.teams = np.append(self.teams, teams_relegated_in)
        self.teams = np.array(sorted(self.teams))

        self.p_alpha = np.array([])
        self.q_alpha = np.array([])
        self.p_beta = np.array([])
        self.q_beta = np.array([])
        self.p_gamma = np.array([])
        self.q_gamma = np.array([])
        for i in range(self.NT):
            if self.teams[i] in list(teams_df['teams']):
                team_data = teams_df[teams_df['teams']==self.teams[i]]
                w_b = self.w_b
                w3 = self.w3
            elif self.teams[i] in list(teams_promoted_in):
                team_data = self.promoted
                w_b = 1
                w3 = 1
            elif self.teams[i] in list(teams_relegated_in):
                team_data = self.relegated
                w_b = 1
                w3 = 1
                
            self.p_alpha = np.append(self.p_alpha, w_b*float(team_data['p_alpha'].iloc[0] if hasattr(team_data['p_alpha'], 'iloc') else team_data['p_alpha']))
            self.q_alpha = np.append(self.q_alpha, w_b*float(team_data['q_alpha'].iloc[0] if hasattr(team_data['q_alpha'], 'iloc') else team_data['q_alpha']))
            self.p_beta = np.append(self.p_beta, w_b*float(team_data['p_beta'].iloc[0] if hasattr(team_data['p_beta'], 'iloc') else team_data['p_beta']))
            self.q_beta = np.append(self.q_beta, w_b*float(team_data['q_beta'].iloc[0] if hasattr(team_data['q_beta'], 'iloc') else team_data['q_beta']))
            self.p_gamma = np.append(self.p_gamma, w3*float(team_data['p_gamma'].iloc[0] if hasattr(team_data['p_gamma'], 'iloc') else team_data['p_gamma']))
            self.q_gamma = np.append(self.q_gamma, w3*float(team_data['q_gamma'].iloc[0] if hasattr(team_data['q_gamma'], 'iloc') else team_data['q_gamma']))
            
        self.alpha_hat = (self.p_alpha-1)/self.q_alpha
        self.beta_hat = (self.p_beta-1)/self.q_beta
        self.gamma_hat = (self.p_gamma-1)/self.q_gamma
        
    def start_next_season(self, season, league_below='E1', league_below_2='E2'):
        """Advance the model into `season` before that division's own file exists.

        football-data.co.uk publishes a division's CSV only once it kicks off, and
        the Premier League lags the EFL by a week or two, so there is a window each
        August where the new season has started but there are no matches to train
        on. The new line-up is already implied by the divisions below, though:
        whoever went down turns up in the feeder division, and whoever came up is
        missing from both the feeder division and the one below that. Applying the
        promoted/relegated priors now means the new sides can be predicted straight
        away instead of erroring as unknown teams.

        Returns (teams_out, promoted_in) if the model was advanced, else None.
        """
        below_now = read_season(season, league_below)
        below_before = read_season(season - 1, league_below)
        below_2_now = read_season(season, league_below_2)
        if below_now.empty or below_before.empty or below_2_now.empty:
            print('Cannot roll forward into ' + str(season) +
                  ': need ' + str(league_below) + ' for ' + str(season - 1) + ' and ' +
                  str(season) + ' plus ' + str(league_below_2) + ' for ' + str(season))
            return None

        now_below = set(season_teams(below_now))
        teams_out = sorted(set(self.teams) & now_below)
        promoted_in = sorted(set(season_teams(below_before)) - now_below - set(season_teams(below_2_now)))

        # new_season rebuilds the parameter arrays over a fixed roster size, so a
        # lopsided swap would silently corrupt them. Bail out instead.
        if not teams_out or len(teams_out) != len(promoted_in):
            print('Cannot roll forward into ' + str(season) + ': inferred ' +
                  str(len(teams_out)) + ' out / ' + str(len(promoted_in)) + ' in ' +
                  '(' + str(teams_out) + ' / ' + str(promoted_in) + ')')
            return None

        self.new_season(teams_out, promoted_in)
        self.rolled_forward_to = season
        print('Rolled forward into ' + str(season) + ': out ' + str(teams_out) +
              ', in ' + str(promoted_in))
        return teams_out, promoted_in

    def train_all(self, league_str, league_below=None, league_above=None, SEA = list(range(1996, 2021))):
        SEA = available_seasons(SEA, league_str)
        self.seasons_trained = list(SEA)
        
        data = read_football_data(season_file(SEA[0], league_str))
        teams = season_teams(data)

        self.teams = teams
        self.NT = len(teams)

        if league_below:
            data_below = read_season(SEA[0], league_below)
            teams_below = season_teams(data_below)

        if league_above:
            data_above = read_season(SEA[0], league_above)
            teams_above = season_teams(data_above)

        print('Season: ' + str(SEA[0]), end="\r")
        self.initialise(teams)
        self.train(data)
        matches = data.shape[0]
        seasons = [SEA[0]]*matches
        promoted_in=None
        relegated_in=None
        for i in range(1, len(SEA)):
            print('Season: ' + str(SEA[i]), end="\r")
            old_data = data
            old_teams = teams
            data = read_football_data(season_file(SEA[i], league_str))
            matches = data.shape[0]
            seasons = np.append(seasons, [SEA[i]]*matches)
            teams = season_teams(data)
            teams_out = list(set(old_teams) - set(teams))

            if league_below:
                old_data_below = data_below
                old_teams_below = teams_below
                data_below = read_season(SEA[i], league_below)
                teams_below = season_teams(data_below)

            if league_above:
                old_data_above = data_above
                old_teams_above = teams_above
                data_above = read_season(SEA[i], league_above)
                teams_above = season_teams(data_above)

            if league_below:
                promoted_in =  sorted(list(set(old_teams_below) & set(teams)))
            if league_above:
                relegated_in = sorted(list(set(old_teams_above) & set(teams)))

            if not (league_below or league_above):
                promoted_in =  sorted(list(set(teams) - set(old_teams)))

            self.new_season(teams_out, promoted_in, relegated_in)
            self.train(data)
        self.trained_data.insert(0, 'SEA', seasons)
        print('Training Complete')
    
    def betting_odds(self, league_str, SEA=list(range(2006, 2022))):
        NS = []
        for i in SEA:
            NS.append('BettingData/'+str(i)+league_str+'.csv')
        frames = []
        for i in range(len(NS)):
            newdata = pd.read_csv(NS[i])
            if SEA[i] >= 2020:
                columns = list(newdata.columns)
                newdata.columns = [re.sub('Max', 'BbMx', col) for col in columns]
            newdata['Date'] = pd.to_datetime(newdata['Date'], dayfirst=True)
            newdata.insert(1, 'SEA', SEA[i])
            frames.append(newdata[['Div', 'SEA', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'BbMxH', 'BbMxD', 'BbMxA']])
        data = pd.concat(frames, ignore_index=True)
        data['BetHome'] = 1/data['BbMxH']
        data['BetDraw'] = 1/data['BbMxD']
        data['BetAway'] = 1/data['BbMxA']
        return data
        
def remove_end_commas(string):
    if len(string)==0:
        string = ''
    elif string[-1] != ',':
        pass
    else:
        string = remove_end_commas(string[0:(len(string)-1)])
    return string

def read_football_data(file):
    data = []
    DataFile = open(file, "r", encoding='latin-1')
    i=0
    while True:
        i += 1
        newline = DataFile.readline()
        newline = newline.rstrip()
        newline = remove_end_commas(newline)
        if len(newline) < 4:
            break
        readData = newline.split(",")
        if i==1:
            columns = np.array(readData)
        else:
            data.append(readData)
    DataFile.close()
    ftr_pos = int(np.where(columns=='FTR')[0])
    output = pd.DataFrame(data).iloc[:, :(ftr_pos+1)]
    output.columns=columns[:(ftr_pos+1)]
    return output[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]

def read_betting_data(file):
    data = []
    DataFile = open(file, "r", encoding='latin-1')
    i=0
    while True:
        i += 1
        # Read new line
        newline = DataFile.readline()
        newline = newline.rstrip()
        # If line empty, stop
        newline = remove_end_commas(newline)
        if len(newline) < 4:
            break
        #print(newline)
        #print(len(newline))
        #split comma seperated values into list
        readData = newline.split(",")
        if i==1:
            columns = np.array(readData)

        else:
            #append data
            data.append(readData)

    DataFile.close()
    ftr_pos = int(np.where(columns=='FTR')[0])
    output = pd.DataFrame(data)
    output.columns = columns
    return output

def get_points_matrix(home, away):
    points = np.zeros((9,9))
    # get result
    if home>away:
        result='Home'
    elif home==away:
        result='Draw'
    elif home<away:
        result='Away'
    # add all result points
    if result=='Home':
        for i in range(9):
            for j in range(i):
                points[i, j]=1
    if result=='Draw':
        for i in range(9):
                points[i, i]=1
    if result=='Away':
        for i in range(9):
            for j in range(i+1, 9):
                points[i, j]=1
    # add exact points
    points[home, away]=3
    # add close points 
    points[home+1, away]=1.5
    points[home, away+1]=1.5
    points[home+1, away+1]=1.5
    if away>0:
        points[home, away-1]=1.5
    if home>0:
        points[home-1, away]=1.5
    if (home>0) & (away>0):
        points[home-1, away-1]=1.5

    if result=='Home':
        for i in range(9):
            for j in range(i, 9):
                points[i, j]=0
    if result=='Draw':
        for i in range(9):
            for j in range(i):
                points[i, j]=0
        for i in range(9):
            for j in range(i+1, 9):
                points[i, j]=0
    if result=='Away':
        for i in range(9):
            for j in range(i+1):
                points[i, j]=0
                
    return points