#-----------------------------------------------Imports-----------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Charchement de l'exel avec les ratios


#------------------------------------------Créer les series en fonction de leur taille et du ratio voulu----------------
def create_series(size,ratio):
    df = pd.read_excel('datas_ratios.xlsx', index_col=None)
    rows = 0
    keep_going = 1
    # start with empty row
    for i in range(size):
        df["liq_ratio - %d" % (i + 1)] = 0
    for i in range(size):
        df["cap_ratio - %d" % (i + 1)] = 0
    df['to_drop'] = 0

    for i in df.index:

        if i >= size:

            if df.loc[i, 'id'] == df.loc[i - size, 'id'] and df.loc[i - size, 'balancesheet'] == 1:

                for j in range(size):
                    df.loc[i, "liq_ratio - %d" % (j + 1)] = df.loc[i - (j + 1), "liq_ratio%d" % ratio]
                    df.loc[i, "cap_ratio - %d" % (j + 1)] = df.loc[i - (j + 1), 'cap_ratio']
            else:
                df.loc[i, 'to_drop'] = 1
        else:
            df.loc[i, 'to_drop'] = 1

    for i in df.index:
        if df.loc[i, 'to_drop'] == 1:
            df.drop(i, inplace=True)
    df = df.drop(df.columns[0], axis=1)
    df.to_excel("datas_series.xlsx")

    return df

