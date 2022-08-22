import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from linearmodels import IV2SLS
import re
import dataframe_image as dfi
from tqdm import tqdm
import telegram_send
import time

time_start = time.time()

# 0 --- Main settings
# tel_config = '.conf'

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


def reg_ols(endog, exog, data, covtype):
    # Preliminaries
    d = data.copy()
    eqn = endog + ' ~ ' + ' + '.join(exog)
    # Estimation
    mod = smf.ols(eqn, data=d)
    est = mod.fit(cov_type=covtype_ols)
    print(est.summary2())
    # Arranging output
    beta = pd.DataFrame(est.params, columns=['Coefficients'])
    ci = pd.DataFrame(est.conf_int()).rename(columns={0: 'Lower', 1: 'Upper'})
    pval = pd.DataFrame(est.pvalues, columns=['P-Values'])
    rsq = pd.DataFrame(['Adjusted R-Squared', est.rsquared_adj])
    bayesic = pd.DataFrame(['Bayesian Information Criterion', est.bic])
    samplesize = pd.DataFrame(['Number of Observations', est.nobs])
    ssresid = pd.DataFrame(['Sum of Squared Residuals', est.ssr])
    tab_beta = pd.concat([beta, ci, pval], axis=1)
    tab_stats = pd.concat([rsq, bayesic, samplesize, ssresid], axis=1).transpose()
    # Output
    return est, tab_beta, tab_stats


def reg_iv2sls(lhs_endog, rhs_endog, included_inst, excluded_inst, data, covtype):
    # Preliminaries
    d = data.copy()
    eqn = lhs_endog + ' ~ 1 + ' + ' + '.join(included_inst) + ' + [' + ' + '.join(rhs_endog) + ' ~ ' + ' + '.join(excluded_inst) + ']'
    # Estimation
    mod = IV2SLS.from_formula(eqn, data=d)
    est = mod.fit(cov_type=covtype)
    print(est.summary)
    # Arranging Output
    # 2nd stage
    beta = pd.DataFrame(est.params).rename(columns={'parameter': 'Coefficients'})
    ci = pd.DataFrame(est.conf_int()).rename(columns={'lower': 'Lower', 'upper': 'Upper'})
    pval = pd.DataFrame(est.pvalues).rename(columns={'pvalue': 'P-Values'})
    rsq = pd.DataFrame(['Adjusted R-Squared', est.rsquared_adj])
    fstat = pd.DataFrame(['First Stage F-Stat', est.first_stage.diagnostics['f.stat'].iloc[0]])
    samplesize = pd.DataFrame(['Number of Observations', est.nobs])
    ssresid = pd.DataFrame(['Sum of Squared Residuals', est.resid_ss])
    tab_beta = pd.concat([beta, ci, pval], axis=1)
    tab_stats = pd.concat([fstat, rsq, samplesize, ssresid], axis=1).transpose()
    # Output
    return est, tab_beta, tab_stats


def ols_histdecomp(ols_results, data):
    # Preliminaries
    d = data.copy()
    d['Intercept'] = 1  # new column of ones
    # Match columns
    beta = ols_results.params
    beta_labels = beta.index.tolist()
    # Compute contributions for all obs
    for x in beta_labels:
        d[x + '_cont'] = d[x] * beta.loc[x]
    d['residuals'] = ols_results.resid
    # Reduce to just input and output data
    col_keep = [x + '_cont' for x in beta_labels] + ['residuals'] + [ols_results.model.endog_names]
    d = d[col_keep]
    # Output
    return d


def iv_histdecomp(iv_results, data):
    # Preliminaries
    d = data.copy()
    d['Intercept'] = 1  # new column of ones
    # Match columns
    beta = iv_results.params
    beta_labels = beta.index.tolist()
    # Compute contributions for all obs
    for x in beta_labels:
        d[x + '_cont'] = d[x] * beta.loc[x]
    d['residuals'] = iv_results.resids
    # Reduce to just input and output data
    col_keep = [x + '_cont' for x in beta_labels] + ['residuals'] + iv_results.model.dependent.cols
    d = d[col_keep]
    # Output
    return d


# II --- Data
df = pd.read_csv('LLTO_CrossSect_Data.txt', sep='|')
# Retail + grocery mobility
df['retgromob'] = (df['retailmob'] + df['grocerymob']) / 2
# Rescale mobility to 0 to 100
col_mob = ['retailmob', 'grocerymob', 'workmob', 'retgromob']
for i in col_mob:
    df[i] = df[i] + 100
# Calculate average GDP during the period (100 = 2019Q4)
df['rgdpsa'] = 100 * (1 + df['rgdpqoqsa'] / 100)
# log-levels
col_og = ['rgdpsa', 'rgdpqoqsa', 'coviddeaths', 'fullvax', 'econsupport', 'gci',
          'gci_institutions', 'population', 'grocerymob', 'retailmob', 'workmob',
          'stringency', 'retgromob']  # exclude real gdp growth
col_nologs = ['rgdpqoqsa']  # interpret as impact in percentage points of QoQSA growth
# col_transform = col_og.copy()
# for i in col_nologs:
#     col_transform.remove(i)
# for i in col_transform:
#     df[i] = np.log(df[i])
# Drop vax = log(0) countries
# df = df[~(df['country'] == 'Taiwan')]

# III.A --- OLS
# Setup
lhs = 'rgdpqoqsa'
exog_ols = ['coviddeaths', 'stringency', 'econsupport', 'fullvax']
eqn_ols = lhs + ' ~ ' + ' + '.join(exog_ols)
covtype_ols = 'HC0'
# Estimation
est_ols, beta_ols, stats_ols = reg_ols(
    endog=lhs,
    exog=exog_ols,
    data=df,
    covtype=covtype_ols
)
stats_ols = stats_ols.rename(columns={0: 'Stat'}).set_index('Stat')
# Historical Decomposition
hd_ols = ols_histdecomp(ols_results=est_ols, data=df)
hd_ols = hd_ols.agg('mean')

# III.B --- IV2SLS
# Setup
inc_inst = ['coviddeaths', 'stringency', 'fullvax']
rhs_endog = ['econsupport']
exc_inst = ['gci_institutions']
covtype_iv = 'unadjusted'
# Estimation
est_iv, beta_iv, stats_iv = reg_iv2sls(
    lhs_endog=lhs,
    rhs_endog=rhs_endog,
    included_inst=inc_inst,
    excluded_inst=exc_inst,
    data=df,
    covtype=covtype_iv
)
stats_iv = stats_iv.rename(columns={0: 'Stat'}).set_index('Stat')
# Historical Decomposition
hd_iv = iv_histdecomp(iv_results=est_iv, data=df)
hd_iv = hd_iv.agg('mean')

# III.X --- Consolidate
beta_ols.columns = pd.MultiIndex.from_product([['OLS'], beta_ols.columns])
beta_iv.columns = pd.MultiIndex.from_product([['IV2SLS'], beta_iv.columns])
beta_consol = pd.concat([beta_ols, beta_iv], axis=1)

stats_consol = pd.concat([pd.DataFrame(stats_ols).rename(columns={1: 'OLS'}),
                          pd.DataFrame(stats_iv).rename(columns={1: 'IV2SLS'})], axis=1)

hd_consol = pd.concat([pd.DataFrame(hd_ols, columns=['OLS']), pd.DataFrame(hd_iv, columns=['IV2SLS'])], axis=1)
hd_consol = hd_consol.round(4)

dfi.export(beta_consol, 'Output/LLTO_CrossSect_Est_BetaConsol.png')
dfi.export(stats_consol, 'Output/LLTO_CrossSect_Est_StatsConsol.png')
dfi.export(hd_consol, 'Output/LLTO_CrossSect_Est_HDConsol.png')

# telsendimg(conf=tel_config,
#            path='Output/LLTO_CrossSect_Est_BetaConsol.png',
#            cap='LLTO_CrossSect_Est_BetaConsol')
# telsendimg(conf=tel_config,
#            path='Output/LLTO_CrossSect_Est_StatsConsol.png',
#            cap='LLTO_CrossSect_Est_StatsConsol')
# telsendimg(conf=tel_config,
#            path='Output/LLTO_CrossSect_Est_HDConsol.png',
#            cap='LLTO_CrossSect_Est_HDConsol')

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
    title='Historical Decomposition of Average Real GDP QoQSA (Cross-Sectional Data)',
    yaxis_title='Percentage Points',
    plot_bgcolor='white',
    height=768,
    width=1366,
    font=dict(color='black', size=12),
    uniformtext=dict(mode='show', minsize=10)
)
fig.write_image('Output/LLTO_CrossSect_Est_HDChart.png')
fig.write_html('Output/LLTO_CrossSect_Est_HDChart.html')
# telsendimg(conf=tel_config,
#            path='Output/LLTO_CrossSect_Est_HDChart.png',
#            cap='LLTO_CrossSect_Est_HDChart')
# No 'Others'
del hd_consol['Intercept']
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
    title='Historical Decomposition of Average Real GDP QoQSA (Cross-Sectional Data); Excluding Intercept, and Residuals',
    yaxis_title='Percentage Points',
    plot_bgcolor='white',
    height=768,
    width=1366,
    font=dict(color='black', size=12),
    uniformtext=dict(mode='show', minsize=10)
)
fig.write_image('Output/LLLTO_CrossSect_Est_HDChart_NoOthers.png')
fig.write_html('Output/LLLTO_CrossSect_Est_HDChart_NoOthers.html')
# telsendimg(conf=tel_config,
#            path='Output/LLLTO_CrossSect_Est_HDChart_NoOthers.png',
#            cap='LLLTO_CrossSect_Est_HDChart_NoOthers')

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
