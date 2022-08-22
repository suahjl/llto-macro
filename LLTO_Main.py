import pandas as pd
import numpy as np
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


# II --- Flow
# import LLTO_Data
import LLTO_StylisedPlots
import LLTO_CrossSect_Est
import LLTO_Quarterly_Est

# End
text_time = '\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----'
print(text_time)
# telsendmsg(conf=tel_config,
#            msg=text_time)

