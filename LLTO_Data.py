from ceic_api_client.pyceic import Ceic
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
import re
from tqdm import tqdm
import telegram_send
import time

time_start = time.time()

# 0 --- Preliminaries
seriesid_labels = pd.read_csv(
    'seriesid.csv',
    dtype='str',
    # usecols=['gci_institutions']
)  # take in everything as strings

seriesid_labels = seriesid_labels.replace({'\([^()]*\)': ''}, regex=True)  # remove everything inside parentheses
seriesid_labels = seriesid_labels.replace({' ': ''}, regex=True)  # remove spaces

tminus = date(2019, 1, 1)
t0 = date(2020, 1, 1)  # Start of pandemic mode
tfin = date(2021, 12, 31)  # Omicron coincided with effective end of pandemic mode

col_arrangement1 = ['country', 'date'] + list(seriesid_labels.columns)
col_arrangement2 = ['country', 'quarter'] + list(seriesid_labels.columns)

col_daily = ['coviddeaths', 'fullvax', 'econsupport',
             'grocerymob', 'retailmob', 'workmob', 'stringency']
col_quarterly = ['rgdpqoqsa']
col_annual = ['gci', 'gci_institutions', 'population']

col_sum = ['coviddeaths', 'fullvax']
col_avg = ['econsupport',
           'gci', 'gci_institutions',
           'population',
           'grocerymob', 'retailmob', 'workmob',
           'stringency',
           'rgdpqoqsa']

col_mob = ['grocerymob', 'retailmob', 'workmob']
col_capita = ['coviddeaths', 'fullvax']

col_strictmissing = ['grocerymob', 'retailmob', 'workmob',
                     'stringency',
                     'rgdpqoqsa', 'rgdpsa'
                     'coviddeaths', 'fullvax',
                     'econsupport']

Ceic.login("", "")  # Use own CEIC ID and password

# I --- Functions


def ceic2pandas(input, t_start, t_end):  # input should be a list of CEIC Series IDs
    for m in range(len(input)):
        try: input.remove(np.nan)  # brute force remove all np.nans from series ID list
        except: print('no more np.nan')
    k = 1
    for i in tqdm(input):
        series_result = Ceic.series(i, start_date=t_start, end_date=t_end)  # retrieves ceicseries
        y = series_result.data
        name = y[0].metadata.country.name  # retrieves country name
        longname = y[0].metadata.name # retrieves CEIC series name
        time_points_dict = dict((tp.date, tp.value) for tp in y[0].time_points)  # this is a list of 1 dictionary,
        series = pd.Series(time_points_dict)  # convert into pandas series indexed to timepoints
        if k == 1:
            frame_consol = pd.DataFrame(series)
            frame_consol['country'] = name
            if re.search('Hong Kong', longname):
                frame_consol['country'] = 'Hong Kong'
            if re.search('Macau', longname):
                frame_consol['country'] = 'Macau'
            frame_consol = frame_consol.reset_index(drop=False).rename(columns={'index': 'date'})
        elif k > 1:
            frame = pd.DataFrame(series)
            frame['country'] = name
            if re.search('Hong Kong', longname):
                frame['country'] = 'Hong Kong'
            if re.search('Macau', longname):
                frame['country'] = 'Macau'
            frame = frame.reset_index(drop=False).rename(columns={'index': 'date'})
            frame_consol = pd.concat([frame_consol, frame], axis=0) # top-bottom concat
        elif k < 1:
            raise NotImplementedError
        k += 1
    frame_consol = frame_consol.reset_index(drop=True)  # avoid repeating indices
    return frame_consol


def telsendimg(path='', conf='', cap=''):
    with open(path, 'rb') as f:
        telegram_send.send(conf=conf,
                           images=[f],
                           captions=[cap])



def telsendfiles(path='', conf='', cap=''):
    with open(path, 'rb') as f:
        telegram_send.send(conf=conf,
                           files=[f],
                           captions=[cap])


def telsendmsg(conf='', msg=''):
    telegram_send.send(conf=conf,
                       messages=[msg])


# II --- Generate panel data
# Download and concat from CEIC
count = 1
for x in list(seriesid_labels.columns):  # which variable (no tqdm to avoid repeating bars)
    input = list(seriesid_labels[x])  # generate list of series IDs for the variable of interest
    if count == 1:
        df = ceic2pandas(input=input, t_start=tminus, t_end=tfin)  # single variable panel
        df = df.rename(columns={0: x})  # rename default column name
        if x in (col_quarterly + col_annual):
            df.loc[pd.to_datetime(df['date']).dt.month.isin([1, 3, 5, 7, 8, 10, 12]),
                   'date'] = df['date'] + timedelta(days=30)
            df.loc[pd.to_datetime(df['date']).dt.month.isin([4, 6, 9, 11]),
                   'date'] = df['date'] + timedelta(days=29)
            df.loc[(pd.to_datetime(df['date']).dt.month.isin([2])) & (pd.to_datetime(df['date']).dt.year.isin([2020])),
                   'date'] = df['date'] + timedelta(days=28)  # leap year
            df.loc[(pd.to_datetime(df['date']).dt.month.isin([2])) & ~(pd.to_datetime(df['date']).dt.year.isin([2020])),
                   'date'] = df['date'] + timedelta(days=27)  # leap year
    elif count > 1:
        d = ceic2pandas(input=input, t_start=tminus, t_end=tfin)  # single variable panel
        d = d.rename(columns={0: x})  # rename default column name
        if x in (col_quarterly + col_annual):
            d.loc[pd.to_datetime(d['date']).dt.month.isin([1, 3, 5, 7, 8, 10, 12]),
                   'date'] = d['date'] + timedelta(days=30)
            d.loc[pd.to_datetime(d['date']).dt.month.isin([4, 6, 9, 11]),
                   'date'] = d['date'] + timedelta(days=29)
            d.loc[(pd.to_datetime(d['date']).dt.month.isin([2])) & (pd.to_datetime(d['date']).dt.year.isin([2020])),
                   'date'] = d['date'] + timedelta(days=28)  # leap year
            d.loc[(pd.to_datetime(d['date']).dt.month.isin([2])) & ~(pd.to_datetime(d['date']).dt.year.isin([2020])),
                   'date'] = d['date'] + timedelta(days=27)  # leap year
        df = df.merge(d, how='outer', on=['country', 'date'])  # left-right merge (new column)
    elif count < 1:
        raise NotImplementedError
    count += 1
# Chronological order by country
df = df.sort_values(by=['country', 'date'], ascending=[True, True])
# Forward and backward fill annual columns that end early (gci and pop)
for i in tqdm(col_annual):
    df[i] = df.groupby('country')[i].fillna(method='ffill')
    df[i] = df.groupby('country')[i].fillna(method='bfill')
# Backward fill quarterly data
for i in tqdm(col_quarterly):
    df[i] = df.groupby('country')[i].fillna(method='bfill')
# Backward fill mobility data (for full 1Q 2020 data)
for i in tqdm(col_mob):
    df[i] = df.groupby('country')[i].fillna(method='bfill')
# Replace vax and deaths with per 1k population
for i in tqdm(col_capita):
    df[i] = df[i] / (1000 * df['population'])  # pop = million
    df.loc[df[i].isna(), i] = 0  # if zero deaths

# III.A --- Generate cross-section data set
df_cx = df.copy()
# Timebound to effective COVID pandemic
df_cx = df_cx[(df_cx['date'] >= t0) & (df_cx['date'] <= tfin)]
# Arrangement
df_cx = df_cx[col_arrangement1]
# Ensure no missing values for strictly balanced columns
for i in tqdm(col_strictmissing + ['gci_institutions']):
    for j in list(df_cx['country'].unique()):
        df_cx.loc[(df_cx['country'] == j) & (df_cx[i].isna()), '_missing'] = 1
        df_cx.loc[(df_cx['country'] == j) & ~(df_cx[i].isna()), '_missing'] = 0
        country_has_missing = df_cx.loc[df_cx['country'] == j, '_missing'].max()
        df_cx.loc[df_cx['country'] == j, '_missing'] = country_has_missing  # now 1 = drop
    df_cx = df_cx[df_cx['_missing'] == 0]  # keep only countries without missing 'strict' columns
del df_cx['_missing']
# Collapse into cross section
df_cx_sum = df_cx.groupby('country')[col_sum].agg('sum').reset_index(drop=False)
df_cx_avg = df_cx.groupby('country')[col_avg].agg('mean').reset_index(drop=False)
df_cx = df_cx_sum.merge(df_cx_avg, how='outer', on='country')  # left-right merge

# III.B --- Generate quarterly panel data set
df_q = df.copy()
# Timebound to effective COVID pandemic
df_q = df_q[(df_q['date'] >= t0) & (df_q['date'] <= tfin)]
# Arrangement
df_q = df_q[col_arrangement1]
# Time column
df_q['quarter'] = pd.to_datetime(df_q['date']).dt.to_period('Q')
df_q = df_q[col_arrangement2]
# Ensure no missing values for strictly balanced columns
for i in tqdm(col_strictmissing):
    for j in list(df_q['country'].unique()):
        df_q.loc[(df_q['country'] == j) & (df_q[i].isna()), '_missing'] = 1
        df_q.loc[(df_q['country'] == j) & ~(df_q[i].isna()), '_missing'] = 0
        country_has_missing = df_q.loc[df_q['country'] == j, '_missing'].max()
        df_q.loc[df_q['country'] == j, '_missing'] = country_has_missing  # now 1 = drop
    df_q = df_q[df_q['_missing'] == 0]  # keep only countries without missing 'strict' columns
del df_q['_missing']
# Collapse into quarterly data
df_q_sum = df_q.groupby(['country', 'quarter'])[col_sum].agg('sum').reset_index(drop=False)
df_q_avg = df_q.groupby(['country', 'quarter'])[col_avg].agg('mean').reset_index(drop=False)
df_q = df_q_sum.merge(df_q_avg, on=['country', 'quarter'], how='outer')  # left-right merge

# IV --- Export
df_cx.to_csv('LLTO_CrossSect_Data.txt', sep='|', index=False)
telsendfiles(path='LLTO_CrossSect_Data.txt',
             conf='EcMetrics_Config_GeneralFlow.conf',
             cap='LLTO_CrossSect_Data')

df_q.to_csv('LLTO_Quarterly_Data.txt', sep='|', index=False)
telsendfiles(path='LLTO_Quarterly_Data.txt',
             conf='EcMetrics_Config_GeneralFlow.conf',
             cap='LLTO_Quarterly_Data')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
