#!/usr/bin/env python
import io
import os
import time

import pandas as pd
import requests

from Helper import read_football_data, read_betting_data


def fetch_csv(url, encoding='latin-1'):
    r = requests.get(url, allow_redirects=True, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.BytesIO(r.content.lstrip(b'\xef\xbb\xbf')), encoding=encoding)


def download_match_data(year_range=range(1996, 2027), leagues=None, output_dir='AutoData'):
    if leagues is None:
        leagues = ['E0']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    url = 'https://www.football-data.co.uk/mmz4281/'
    t0 = time.time()
    for year in year_range:
        prev_year = str(year - 1)
        year_str = str(year)
        year_string = prev_year[2:] + year_str[2:]
        print(year_string, end='\r')
        for league in leagues:
            full_url = url + year_string + '/' + league + '.csv'
            try:
                r = requests.get(full_url, allow_redirects=True, timeout=30)
                if r.status_code != 200:
                    print(f'Skipping {year_string}/{league}: HTTP {r.status_code}')
                    continue
                filepath = os.path.join(output_dir, year_str + league + '.csv')
                open(filepath, 'wb').write(r.content.lstrip(b'\xef\xbb\xbf'))
                data = read_football_data(filepath)
                data.to_csv(filepath, index=False)
            except Exception as e:
                print(f'Error downloading {year_string}/{league}: {e}')
    print(f'Time elapsed: {round(time.time() - t0, 1)}s')


def download_all_leagues(year_range=range(1996, 2026), output_dir='AutoData'):
    leagues = ['E0', 'E1', 'E2', 'E3', 'SP1', 'SP2', 'SC0', 'SC1', 'D1', 'D2',
               'I1', 'F1', 'N1', 'B1', 'P1', 'T1', 'G1']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    url = 'https://www.football-data.co.uk/mmz4281/'
    t0 = time.time()
    ext = list(leagues)
    for year in year_range:
        prev_year = str(year - 1)
        year_str = str(year)
        year_string = prev_year[2:] + year_str[2:]
        print(year_string, end='\r')
        if year == 1997:
            ext.append('F2')
        if year == 1998:
            ext += ['SC2', 'SC3', 'I2']
        if year == 2006:
            ext.append('EC')
        if year == 2021:
            ext = ['E0', 'E1', 'E2', 'E3', 'SP1', 'SP2', 'SC0', 'D1', 'D2',
                   'I1', 'F1', 'N1', 'B1', 'P1', 'T1', 'G1']
        for league in ext:
            full_url = url + year_string + '/' + league + '.csv'
            try:
                r = requests.get(full_url, allow_redirects=True, timeout=30)
                if r.status_code != 200:
                    print(f'Skipping {year_string}/{league}: HTTP {r.status_code}')
                    continue
                filepath = os.path.join(output_dir, year_str + league + '.csv')
                open(filepath, 'wb').write(r.content.lstrip(b'\xef\xbb\xbf'))
                data = read_football_data(filepath)
                data.to_csv(filepath, index=False)
            except Exception as e:
                print(f'Error downloading {year_string}/{league}: {e}')
    print(f'Time elapsed: {round(time.time() - t0, 1)}s')


def download_betting_data(year_range=range(2006, 2022), leagues=None, output_dir='BettingData'):
    if leagues is None:
        leagues = ['E0']
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    url = 'https://www.football-data.co.uk/mmz4281/'
    for year in year_range:
        prev_year = str(year - 1)
        year_str = str(year)
        year_string = prev_year[2:] + year_str[2:]
        print(year_string, end='\r')
        for league in leagues:
            full_url = url + year_string + '/' + league + '.csv'
            try:
                r = requests.get(full_url, allow_redirects=True, timeout=30)
                if r.status_code != 200:
                    print(f'Skipping {year_string}/{league}: HTTP {r.status_code}')
                    continue
                filepath = os.path.join(output_dir, year_str + league + '.csv')
                open(filepath, 'wb').write(r.content.lstrip(b'\xef\xbb\xbf'))
                data = read_betting_data(filepath)
                data.to_csv(filepath, index=False)
            except Exception as e:
                print(f'Error downloading {year_string}/{league}: {e}')


if __name__ == '__main__':
    download_match_data()
    download_betting_data()
