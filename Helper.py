#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
        data.loc[data['FTHG'] > 5, 'FTHG'] = 5
        data.loc[data['FTAG'] > 5, 'FTAG'] = 5
        # iterate through data
        for i in range(data.shape[0]):
            match = data.iloc[[i]]
            # get indices of home and away sides
            HT = int(np.arange(self.NT)[self.teams==match.iloc[0].loc['HomeTeam']])
            AT = int(np.arange(self.NT)[self.teams==match.iloc[0].loc['AwayTeam']])
            X = int(match['FTHG'])
            Y = int(match['FTAG'])
            
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
        HT = int(np.arange(self.NT)[self.teams==HomeTeam])
        AT = int(np.arange(self.NT)[self.teams==AwayTeam])
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
                
            self.p_alpha = np.append(self.p_alpha, w_b*float(team_data['p_alpha']))
            self.q_alpha = np.append(self.q_alpha, w_b*float(team_data['q_alpha']))
            self.p_beta = np.append(self.p_beta, w_b*float(team_data['p_beta']))
            self.q_beta = np.append(self.q_beta, w_b*float(team_data['q_beta']))
            self.p_gamma = np.append(self.p_gamma, w3* float(team_data['p_gamma']))
            self.q_gamma = np.append(self.q_gamma, w3*float(team_data['q_gamma']))
            
        self.alpha_hat = (self.p_alpha-1)/self.q_alpha
        self.beta_hat = (self.p_beta-1)/self.q_beta
        self.gamma_hat = (self.p_gamma-1)/self.q_gamma
        
    def train_all(self, league_str, league_below=None, league_above=None, SEA = list(range(1996, 2021))):
        NS = []
        NS_below = []
        NS_above = []
        for i in SEA:
            NS.append('AutoData/'+str(i)+league_str+'.csv')
            if league_below:
                NS_below.append('AutoData/'+str(i)+league_below+'.csv')
            if league_above:
                NS_above.append('AutoData/'+str(i)+str(league_above)+'.csv')
        
#         print(NS)
#         print(NS_below)
        data = pd.read_csv(NS[0])
        data = data[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
#        display(data['HomeTeam'])
        teams = np.unique(data['HomeTeam'])
        
        self.teams = teams
        self.NT = len(teams)

        if league_below:
            data_below = pd.read_csv(NS_below[0])
            data_below = data_below[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
            teams_below = np.unique(data_below['HomeTeam'])

        if league_above:
            data_above = pd.read_csv(NS_above[0])
            data_above = data_above[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
            teams_above = np.unique(data_above['HomeTeam'])

        print('Season: ' + str(SEA[0]), end="\r")
        self.initialise(teams)
        self.train(data)
        promoted_in=None
        relegated_in=None
        for i in range(1, len(NS)):
            print('Season: ' + str(SEA[i]), end="\r")
            old_data = data
            old_teams = teams    
            data = pd.read_csv(NS[i], encoding = 'unicode_escape')
            data = data[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
            teams = np.unique(data['HomeTeam'])
            teams_out = list(set(old_teams) - set(teams))

            if league_below:
                old_data_below = data_below
                old_teams_below = teams_below
                data_below = pd.read_csv(NS_below[i], encoding = 'unicode_escape')
                data_below = data_below[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
                teams_below = np.unique(data_below['HomeTeam'])

            if league_above:
                old_data_above = data_above
                old_teams_above = teams_above
                data_above = pd.read_csv(NS_above[i], encoding = 'unicode_escape')
                data_above = data_above[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
                teams_above = np.unique(data_above['HomeTeam'])

            if league_below:
                promoted_in =  sorted(list(set(old_teams_below) & set(teams)))
            if league_above:
                relegated_in = sorted(list(set(old_teams_above) & set(teams)))
            
            if not (league_below or league_above):
                promoted_in =  sorted(list(set(teams) - set(old_teams)))
                

            #print('Teams Out:' + str(teams_out))
            #print('Promoted In:' + str(promoted_in))
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
        data.loc[data['FTHG'] > 5, 'FTHG'] = 5
        data.loc[data['FTAG'] > 5, 'FTAG'] = 5
        
        data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
        
        lambdasH = []
        lambdasA = []
        # iterate through data
        for i in range(data.shape[0]):
            match = data.iloc[[i]]
            # get indices of home and away sides
            HT = int(np.arange(self.NT)[self.teams==match.iloc[0].loc['HomeTeam']])
            AT = int(np.arange(self.NT)[self.teams==match.iloc[0].loc['AwayTeam']])
            # get home and away lambda
            lambdaH = self.alpha_hat[HT]*self.beta_hat[AT]*self.gamma_hat[HT]
            lambdasH.append(lambdaH)
            
            lambdaA = self.alpha_hat[AT]*self.beta_hat[HT]
            lambdasA.append(lambdaA)
            
            X = int(match['FTHG'])
            Y = int(match['FTAG'])
            
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
        
        self.trained_data = self.trained_data.append(data, ignore_index=True) 

    def predict(self, HomeTeam, AwayTeam):
        HT = int(np.arange(self.NT)[self.teams==HomeTeam])
        AT = int(np.arange(self.NT)[self.teams==AwayTeam])
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
                
            self.p_alpha = np.append(self.p_alpha, w_b*float(team_data['p_alpha']))
            self.q_alpha = np.append(self.q_alpha, w_b*float(team_data['q_alpha']))
            self.p_beta = np.append(self.p_beta, w_b*float(team_data['p_beta']))
            self.q_beta = np.append(self.q_beta, w_b*float(team_data['q_beta']))
            self.p_gamma = np.append(self.p_gamma, w3* float(team_data['p_gamma']))
            self.q_gamma = np.append(self.q_gamma, w3*float(team_data['q_gamma']))
            
        self.alpha_hat = (self.p_alpha-1)/self.q_alpha
        self.beta_hat = (self.p_beta-1)/self.q_beta
        self.gamma_hat = (self.p_gamma-1)/self.q_gamma
        
    def train_all(self, league_str, league_below=None, league_above=None, SEA = list(range(1996, 2021))):
        NS = []
        NS_below = []
        NS_above = []
        for i in SEA:
            NS.append('AutoData/'+str(i)+league_str+'.csv')
            if league_below:
                NS_below.append('AutoData/'+str(i)+league_below+'.csv')
            if league_above:
                NS_above.append('AutoData/'+str(i)+str(league_above)+'.csv')
        
        data = pd.read_csv(NS[0])
        data = data[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
        teams = np.unique(data['HomeTeam'])
        
        self.teams = teams
        self.NT = len(teams)

        if league_below:
            data_below = pd.read_csv(NS_below[0])
            data_below = data_below[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
            teams_below = np.unique(data_below['HomeTeam'])

        if league_above:
            data_above = pd.read_csv(NS_above[0])
            data_above = data_above[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
            teams_above = np.unique(data_above['HomeTeam'])

        print('Season: ' + str(SEA[0]), end="\r")
        self.initialise(teams)
        self.train(data)
        matches = data.shape[0]
        seasons = [SEA[0]]*matches
        promoted_in=None
        relegated_in=None
        for i in range(1, len(NS)):
            print('Season: ' + str(SEA[i]), end="\r")
            old_data = data
            old_teams = teams    
            data = pd.read_csv(NS[i], encoding = 'unicode_escape')
            data = data[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
            matches = data.shape[0]
            seasons = np.append(seasons, [SEA[i]]*matches)
            teams = np.unique(data['HomeTeam'])
            teams_out = list(set(old_teams) - set(teams))

            if league_below:
                old_data_below = data_below
                old_teams_below = teams_below
                data_below = pd.read_csv(NS_below[i], encoding = 'unicode_escape')
                data_below = data_below[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
                teams_below = np.unique(data_below['HomeTeam'])

            if league_above:
                old_data_above = data_above
                old_teams_above = teams_above
                data_above = pd.read_csv(NS_above[i], encoding = 'unicode_escape')
                data_above = data_above[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]
                teams_above = np.unique(data_above['HomeTeam'])

            if league_below:
                promoted_in =  sorted(list(set(old_teams_below) & set(teams)))
            if league_above:
                relegated_in = sorted(list(set(old_teams_above) & set(teams)))
                
            if not (league_below or league_above):
                promoted_in =  sorted(list(set(teams) - set(old_teams)))

            #print('Teams Out:' + str(teams_out))
            #print('Promoted In:' + str(promoted_in))
            self.new_season(teams_out, promoted_in, relegated_in)
            self.train(data)
        self.trained_data.insert(0, 'SEA', seasons)
        print('Training Complete')
    
    def betting_odds(self, league_str, SEA=list(range(2006, 2022))):
        NS = []
        for i in SEA:
            NS.append('BettingData/'+str(i)+league_str+'.csv')
        j=0
        for i in range(len(NS)):
            if SEA[i] < 2020:
                newdata = pd.read_csv(NS[i])
            else:
                newdata = pd.read_csv(NS[i])
                columns = list(newdata.columns)
                for j in range(len(columns)):
                    column = columns[j]
                    columns[j] = re.sub('Max', 'BbMx', column)
                newdata.columns = columns
            
            newdata['Date'] = pd.to_datetime(newdata['Date'], dayfirst=True)
            newdata.insert(1, 'SEA', SEA[i])
                
            if j==0:
                data = newdata[['Div', 'SEA', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'BbMxH', 'BbMxD', 'BbMxA']]
            else:
                data = data.append(newdata[['Div', 'SEA', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'BbMxH', 'BbMxD', 'BbMxA']], 
                                  ignore_index=True)
            j += 1
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
        #print(string[0:(len(string)-1)])
        string = remove_end_commas(string[0:(len(string)-1)])
    return string
        
def read_football_data(file):
    data = []
    DataFile = open(file, "r")
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
    output = pd.DataFrame(data).iloc[:, :(ftr_pos+1)]
    output.columns=columns[:(ftr_pos+1)]
    return output[['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]

def remove_end_commas(string):
    if len(string)==0:
        string = ''
    
    elif string[-1] != ',':
        pass
        
    else:
        #print(string[0:(len(string)-1)])
        string = remove_end_commas(string[0:(len(string)-1)])
    return string

def read_football_data(file):
    data = []
    DataFile = open(file, "r")
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
    output = pd.DataFrame(data).iloc[:, :(ftr_pos+1)]
    output.columns=columns[:(ftr_pos+1)]
    return output

def read_betting_data(file):
    data = []
    DataFile = open(file, "r")
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