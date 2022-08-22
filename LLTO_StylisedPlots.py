import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.formula.api as smf
import re
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


def scatter_cx_panel(x_label, y_label, data_cx, data_panel, panel_time_col, time_colours):
    # Prelims
    d_cx = data_cx.copy()
    d_p = data_panel.copy()
    fig = make_subplots(rows=1, cols=2)
    # CX chart
    fig.add_trace(
        go.Scatter(
            x=d_cx[x_label],
            y=d_cx[y_label],
            mode='markers',
            marker=dict(color='black', size=8),
            showlegend=False
        ),
        row=1,
        col=1
    )
    eqn_bfit=y_label + ' ~ ' + x_label
    est_bfit = smf.ols(eqn_bfit, data=d_cx).fit()
    pred_bfit = est_bfit.predict(d_cx[x_label])
    d_cx['_pred'] = pred_bfit
    fig.add_trace(
        go.Scatter(
            x=d_cx[x_label],
            y=d_cx['_pred'],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ),
        row=1,
        col=1
    )
    del d_cx['_pred']
    # Panel chart
    for i, colour in zip(list(d_p[panel_time_col].unique()), time_colours):
        d = d_p[d_p[panel_time_col] == i]
        fig.add_trace(
            go.Scatter(
                x=d[x_label],
                y=d[y_label],
                name=i,
                mode='markers',
                marker=dict(color=colour, size=5),
                showlegend=True
            ),
            row=1,
            col=2
        )
        eqn_bfit=y_label + ' ~ ' + x_label
        est_bfit = smf.ols(eqn_bfit, data=d).fit()
        pred_bfit = est_bfit.predict(d[x_label])
        d['_pred'] = pred_bfit
        fig.add_trace(
            go.Scatter(
                x=d[x_label],
                y=d['_pred'],
                mode='lines',
                line=dict(color=colour, width=1.5),
                showlegend=False
            ),
            row=1,
            col=2
        )
    # Figure settings
    fig.update_layout(
        title='Scatter Plot of ' + y_label + ' Against ' + x_label,
        xaxis_title=x_label,
        yaxis_title=y_label,
        plot_bgcolor='white',
        font=dict(color='black'),
        height=768,
        width=1366,
    )
    # Output
    return fig


# II.A --- CX Data
df_cx = pd.read_csv('LLTO_CrossSect_Data.txt', sep='|')
# Retail + grocery mobility
df_cx['retgromob'] = (df_cx['retailmob'] + df_cx['grocerymob']) / 2
# Rescale mobility to 0 to 100
col_mob = ['retailmob', 'grocerymob', 'workmob', 'retgromob']
for i in col_mob:
    df_cx[i] = df_cx[i] + 100
# Calculate average GDP during the period (100 = 2019Q4)
df_cx['rgdpsa'] = 100 * (1 + df_cx['rgdpqoqsa'] / 100)
# log-levels
# col_og = ['rgdpsa', 'rgdpqoqsa', 'coviddeaths', 'fullvax', 'econsupport', 'gci',
#           'gci_institutions', 'population', 'grocerymob', 'retailmob', 'workmob',
#           'stringency', 'retgromob']  # exclude real gdp growth
# col_nologs = ['rgdpqoqsa']  # interpret as impact in percentage points of QoQSA growth
# col_transform = col_og.copy()
# for i in col_nologs:
#     col_transform.remove(i)
# for i in col_transform:
#     df_cx[i] = np.log(df_cx[i])
# Drop vax = log(0) countries
# df_cx = df_cx[~(df_cx['country'] == 'Taiwan')]

# II.B --- Panel Data
df_q = pd.read_csv('LLTO_Quarterly_Data.txt', sep='|')
# Retail + grocery mobility
df_q['retgromob'] = (df_q['retailmob'] + df_q['grocerymob']) / 2
# Rescale mobility to 0 to 100
col_mob = ['retailmob', 'grocerymob', 'workmob', 'retgromob']
for i in col_mob:
    df_q[i] = df_q[i] + 100
# Back out real GDP
df_q.loc[df_q['quarter'] == t_start, 'rgdpsa'] = 100 + (1 + (df_q['rgdpqoqsa'] / 100))
run = 1
while run <= len(df_q['quarter'].unique()):
    df_q.loc[df_q['rgdpsa'].isna(), 'rgdpsa'] = df_q['rgdpsa'].shift(1) + (1 + (df_q['rgdpqoqsa'] / 100))
    run += 1
# log-levels
# col_og = ['rgdpsa', 'rgdpqoqsa', 'coviddeaths', 'fullvax', 'econsupport', 'gci',
#           'gci_institutions', 'population', 'grocerymob', 'retailmob', 'workmob',
#           'stringency', 'retgromob']  # exclude real gdp growth
# col_nologs = ['rgdpqoqsa']  # interpret as impact in percentage points of QoQSA growth + avoid log(0)
# col_transform = col_og.copy()
# for i in col_nologs:
#     col_transform.remove(i)
# for i in col_transform:
#     df_q[i] = np.log(df_q[i])

# III --- Plot
scatter_xy_combos = [['coviddeaths','rgdpqoqsa'],
                     ['stringency', 'rgdpqoqsa'],
                     ['coviddeaths', 'stringency'],
                     ['stringency', 'econsupport'],
                     ['fullvax', 'coviddeaths'],
                     ['fullvax', 'rgdpqoqsa']]
list_panel_colour = ['gray', 'lightcoral', 'blue', 'green',
                     'black', 'darkred', 'darkblue', 'darkgreen']
for comb in scatter_xy_combos:
    x = comb[0]
    y = comb[1]
    fig = scatter_cx_panel(
        x_label=x,
        y_label=y,
        data_cx=df_cx,
        data_panel=df_q,
        panel_time_col='quarter',
        time_colours=list_panel_colour
    )
    fig.write_image('Output/LLTO_StylisedPlots_Scatter_' + x + '_' + y +'.png')
    # telsendimg(conf=tel_config,
    #            path='Output/LLTO_StylisedPlots_Scatter_' + x + '_' + y +'.png',
    #            cap='Scatter Plots of ' + y + ' Against ' + x)

# End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
