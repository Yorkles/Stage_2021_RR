#------------------------------------------------Imports----------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------Definition des ratios--------------------------------------------------------
def create_ratios ():
    df = pd.read_excel('Datas.xlsx', index_col=None)
    df_t = pd.DataFrame(columns=['id', 'referenceyear'])
    df_t['referenceyear'] = df['referenceyear']
    df_t['id'] = df['id']
    for i in df.index:
        if np.isnan(df.loc[i, 'tot_assets']):
            df.loc[i, 'tot_assets'] = df.loc[i, 'tot_liabilities_equity']
        if np.isnan(df.loc[i, 'tot_liabilities_equity']):
            df.loc[i, 'tot_liabilities_equity'] = df.loc[i, 'tot_assets']


    # ------------set ratios for every row ----------------------------------------------------------------------------------
    df['cap_ratio'] = (df['capital'] + df['capital_surplus']) / df['tot_liabilities_equity']
    df['liq_ratio1'] = (df['cash'] + df['comm_portfolio']) / df['tot_liabilities_equity']
    df['liq_ratio2'] = (df['tot_deposits']) / df['tot_liabilities_equity']
    df['liq_ratio3'] = (df['short_term_credit']) / df['tot_liabilities_equity']

    df['tbtf'] = 0
    for i in df.index:
        if df.loc[i, 'id'] == 1245 or df.loc[i, 'id'] == 739 or df.loc[i, 'id'] == 2379 or df.loc[i, 'id'] == 671 or \
                df.loc[i, 'id'] == 2375 or df.loc[i, 'id'] == 1387:
            df.loc[i, 'tbtf'] = 1
    df = df.fillna(0)
    df.to_excel("datas_ratios.xlsx")


