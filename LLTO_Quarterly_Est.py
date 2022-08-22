import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
from linearmodels import PanelOLS
import re
import dataframe_image as dfi
from tqdm import tqdm
import telegram_send
import time

time_start = time.time()

# 0 --- Main settings
# tel_config = '.conf'
t_start = '2020Q1'

# I --- Functions


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


def reg_fe(lhs, rhs, time_effects, entity_effects, data, covtype):
    # Preliminaries
    d = data.copy()
    if time_effects & entity_effects:
        eqn_effects = ' + TimeEffects + EntityEffects'
    elif (not time_effects) & entity_effects:
        eqn_effects = ' + EntityEffects'
    elif (not time_effects) & (not entity_effects):
        eqn_effects = ''
    eqn = lhs + ' ~ 1 + ' + ' + '.join(rhs) + eqn_effects
    # Estimation
    mod = PanelOLS.from_formula(eqn, data=d)
    est = mod.fit(cov_type=covtype)
    print(est.summary)
    # Arranging Output
    # 2nd stage
    beta = pd.DataFrame(est.params).rename(columns={'parameter': 'Coefficients'})
    ci = pd.DataFrame(est.conf_int()).rename(columns={'lower': 'Lower', 'upper': 'Upper'})
    pval = pd.DataFrame(est.pvalues).rename(columns={'pvalue': 'P-Values'})
    rsq_overall = pd.DataFrame(['Overall R-Squared', est.rsquared_overall])
    rsq_within = pd.DataFrame(['Within R-Squared', est.rsquared_within])
    rsq_between = pd.DataFrame(['Between R-Squared', est.rsquared_between])
    rsq_inclusive = pd.DataFrame(['Inclusive R-Squared', est.rsquared_inclusive])
    samplesize = pd.DataFrame(['Number of Observations', est.nobs])
    ssresid = pd.DataFrame(['Sum of Squared Residuals', est.resid_ss])
    tab_beta = pd.concat([beta, ci, pval], axis=1)
    tab_stats = pd.concat([rsq_overall, rsq_within, rsq_between, rsq_inclusive,
                           samplesize, ssresid], axis=1).transpose()
    # Output
    return est, tab_beta, tab_stats


def histdecomp(results, data):
    # Preliminaries
    d = data.copy()
    d['Intercept'] = 1  # new column of ones
    d['FixedEffects'] = results.estimated_effects  # absorption (0 if pooled)
    # Match columns
    beta = results.params
    beta_labels = beta.index.tolist()
    # Compute contributions for all obs
    for x in beta_labels:
        d[x + '_cont'] = d[x] * beta.loc[x]
    d['residuals'] = results.resids
    # Reduce to just input and output data
    col_keep = ['FixedEffects'] + [x + '_cont' for x in beta_labels] + ['residuals'] + results.model.dependent.vars
    d = d[col_keep]
    # Output
    return d


# II --- Data
df = pd.read_csv('LLTO_Quarterly_Data.txt', sep='|')
# Retail + grocery mobility
df['retgromob'] = (df['retailmob'] + df['grocerymob']) / 2
# Rescale mobility to 0 to 100
col_mob = ['retailmob', 'grocerymob', 'workmob', 'retgromob']
for i in col_mob:
    df[i] = df[i] + 100
# Back out real GDP
df.loc[df['quarter'] == t_start, 'rgdpsa'] = 100 + (1 + (df['rgdpqoqsa'] / 100))
run = 1
while run <= len(df['quarter'].unique()):
    df.loc[df['rgdpsa'].isna(), 'rgdpsa'] = df['rgdpsa'].shift(1) + (1 + (df['rgdpqoqsa'] / 100))
    run += 1
# log-levels
col_og = ['rgdpsa', 'rgdpqoqsa', 'coviddeaths', 'fullvax', 'econsupport', 'gci',
          'gci_institutions', 'population', 'grocerymob', 'retailmob', 'workmob',
          'stringency', 'retgromob']  # exclude real gdp growth
# col_logs = ['rgdpsa']  # interpret as impact in percentage points of QoQSA growth + avoid log(0)
# for i in col_logs:
#     df[i] = np.log(df[i])
# Time subset
df['quarter'] = pd.to_datetime(df['quarter']).dt.to_period('Q')
df = df[df['quarter'] >= t_start]
# Set time and entity indices
df['numerictime'] = df.groupby('country').cumcount() + 1
df = df.set_index(['country', 'numerictime'])  # outer = entity, inner = time

# III.A --- FE model
# Setup
lhs = 'rgdpqoqsa'
exog = ['coviddeaths', 'stringency', 'econsupport', 'fullvax']
covtype_twfe = 'unadjusted'
# Estimation
# Pooled
est_pooled, beta_pooled, stats_pooled = reg_fe(
    lhs=lhs,
    rhs=exog,
    time_effects=False,
    entity_effects=False,
    data=df,
    covtype=covtype_twfe
)
stats_pooled = stats_pooled.rename(columns={0: 'Stat'}).set_index('Stat')
hd_pooled = histdecomp(results=est_pooled, data=df)
hd_pooled = hd_pooled.agg('mean')
# FE
est_fe, beta_fe, stats_fe = reg_fe(
    lhs=lhs,
    rhs=exog,
    time_effects=False,
    entity_effects=True,
    data=df,
    covtype=covtype_twfe
)
stats_fe = stats_fe.rename(columns={0: 'Stat'}).set_index('Stat')
hd_fe = histdecomp(results=est_fe, data=df)
hd_fe = hd_fe.agg('mean')

# TWFE
est_twfe, beta_twfe, stats_twfe = reg_fe(
    lhs=lhs,
    rhs=exog,
    time_effects=True,
    entity_effects=True,
    data=df,
    covtype=covtype_twfe
)
stats_twfe = stats_twfe.rename(columns={0: 'Stat'}).set_index('Stat')
hd_twfe = histdecomp(results=est_twfe, data=df)
hd_twfe = hd_twfe.agg('mean')

# Consolidate
beta_pooled.columns = pd.MultiIndex.from_product([['Pooled'], beta_pooled.columns])
beta_fe.columns = pd.MultiIndex.from_product([['Fixed Effects'], beta_fe.columns])
beta_twfe.columns = pd.MultiIndex.from_product([['Two-Way Fixed Effects'], beta_twfe.columns])
beta_consol = pd.concat([beta_pooled, beta_fe, beta_twfe], axis=1)

stats_consol = pd.concat([pd.DataFrame(stats_pooled).rename(columns={1: 'Pooled'}),
                          pd.DataFrame(stats_fe).rename(columns={1: 'Fixed Effects'}),
                          pd.DataFrame(stats_twfe).rename(columns={1: 'Two-Way Fixed Effects'})], axis=1)

hd_consol = pd.concat([pd.DataFrame(hd_pooled, columns=['Pooled']),
                       pd.DataFrame(hd_fe, columns=['Fixed Effects']),
                       pd.DataFrame(hd_twfe, columns=['Two-Way Fixed Effects'])], axis=1)
hd_consol = hd_consol.round(4)

dfi.export(beta_consol, 'Output/LLTO_Quarterly_Est_BetaConsol.png')
dfi.export(stats_consol, 'Output/LLTO_Quarterly_Est_StatsConsol.png')
dfi.export(hd_consol, 'Output/LLTO_Quarterly_Est_HDConsol.png')
# telsendimg(conf=tel_config,
#            path='Output/LLTO_Quarterly_Est_BetaConsol.png',
#            cap='LLTO_Quarterly_Est_BetaConsol')
# telsendimg(conf=tel_config,
#            path='Output/LLTO_Quarterly_Est_StatsConsol.png',
#            cap='LLTO_Quarterly_Est_StatsConsol')
# telsendimg(conf=tel_config,
#            path='Output/LLTO_Quarterly_Est_HDConsol.png',
#            cap='LLTO_Quarterly_Est_HDConsol')

# Plot historical decomposition
hd_consol.index = hd_consol.index.str.replace('_cont', '')
hd_consol = hd_consol.transpose()
del hd_consol[lhs]
# Full
fig = go.Figure()
for x in tqdm(hd_consol.columns.tolist()):
    fig.add_trace(
        go.Bar(
            x=hd_consol.index,
            y=hd_consol[x],
            name=x,
            text=hd_consol[x],
            textposition='auto'
        )
    )
fig.update_layout(
    barmode='relative',
    title='Historical Decomposition of Real GDP QoQSA (Quarterly Data)',
    yaxis_title='Percentage Points',
    plot_bgcolor='white',
    height=768,
    width=1366,
    font=dict(color='black', size=12),
    uniformtext=dict(mode='show', minsize=10)
)
fig.write_image('Output/LLTO_Quarterly_Est_HDChart.png')
fig.write_html('Output/LLTO_Quarterly_Est_HDChart.html')
# telsendimg(conf=tel_config,
#            path='Output/LLTO_Quarterly_Est_HDChart.png',
#            cap='LLTO_Quarterly_Est_HDChart')
# No 'Others'
del hd_consol['Intercept']
del hd_consol['FixedEffects']
del hd_consol['residuals']
fig = go.Figure()
for x in tqdm(hd_consol.columns.tolist()):
    fig.add_trace(
        go.Bar(
            x=hd_consol.index,
            y=hd_consol[x],
            name=x,
            text=hd_consol[x],
            textposition='auto'
        )
    )
fig.update_layout(
    barmode='relative',
    title='Historical Decomposition of Real GDP QoQSA (Quarterly Data); Excluding FEs, Intercept, and Residuals',
    yaxis_title='Percentage Points',
    plot_bgcolor='white',
    height=768,
    width=1366,
    font=dict(color='black', size=12),
    uniformtext=dict(mode='show', minsize=10)
)
fig.write_image('Output/LLTO_Quarterly_Est_HDChart_NoOthers.png')
fig.write_html('Output/LLTO_Quarterly_Est_HDChart_NoOthers.html')
# telsendimg(conf=tel_config,
#            path='Output/LLTO_Quarterly_Est_HDChart_NoOthers.png',
#            cap='LLTO_Quarterly_Est_HDChart_NoOthers')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
