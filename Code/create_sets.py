#----------------------------------------------------Imports--------------------------------------------------------
import math
import random
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import sklearn.metrics as mt
from sklearn.model_selection import GridSearchCV
import seaborn as sns

#----------------------------------------Creation des train et test sets------------------------------------------------
def create_sets(size,df,proportion_fail,fail,seed):
    np.random.seed(seed)
    random.seed(seed)
    df['biais'] = 1
    nb_banques = df.nunique().loc['id']
    nb_f1 = df.sum().loc['f1']
    nb_f2 = df.sum().loc['f2']
    nb_f3 = df.sum().loc['f3']
    # --------------------------------------------print statistiques---------------------------------------------------------
    print("sur %d banques" % nb_banques)
    print("%d sont f1" % nb_f1)
    print("%d  sont f2" % nb_f2)
    print("%d  sont f3" % nb_f3)
    # -------------------------------------------split dataset---------------------------------------------------------------
    # here we have to get a beter proportion of failure and also we want have the big bank marked idk how we will do dis
    # also he wanted random series like we could take any series from non failling bank.
    # identification dee sisx grosses banques we want the same amount propr in test and train
    # ajout des tntf
    # creation d'une copie
    df_copy = pd.DataFrame(columns=df.columns)
    # fill
    id_to_drop = 0
    #selectionne une series par grosse banque
    for i in df.index:
        if df.loc[i, 'tbtf'] == 1 and df.loc[i, 'to_drop'] == 0:
            if df.loc[i, 'id'] != id_to_drop:
                df_copy = df_copy.append(df.loc[i])
                id_to_drop = df.loc[i, 'id']
                df.loc[i, 'to_drop'] = 1
            else:
                df.loc[i, 'to_drop'] = 1
    #suppression des grosses banques dans le df d'origine
    for i in df.index:
        if df.loc[i, 'to_drop'] == 1:
            df.drop(i, inplace=True)
    # une iteraion de chaque banque qui fait faillite
    for i in df.index:
        if df.loc[i, "f%d" % fail] == 1 and df.loc[i, 'to_drop'] == 0:
            if df.loc[i, 'id'] != id_to_drop:
                df_copy = df_copy.append(df.loc[i])
                id_to_drop = df.loc[i, 'id']
                df.loc[i, 'to_drop'] = 1
                df.drop(df[df.id == id_to_drop].index, inplace=True)
            else:
                df.loc[i, 'to_drop'] = 1
    for i in df.index:
        if df.loc[i, 'to_drop'] == 1:
            df.drop(i, inplace=True)



# definition de la taille du set pour conserver une proportion voulue
    set_size = math.ceil(0.9 * df_copy.sum().loc["f%d" % fail] * (1/proportion_fail) * (10 / 7))
    while len(df_copy) < set_size:
        i = random.randint(0, len(df) - 1)

        df_copy = df_copy.append(df.iloc[i])
        id_to_drop = df.iloc[i, 1]
        df.drop(df[df.id == id_to_drop].index, inplace=True)
    S = pd.DataFrame()
    for i in range(1, size + 1):
        S["cap_ratio - %d" % i] = df_copy["cap_ratio - %d" % i]
        S["liq_ratio - %d" % i] = df_copy["liq_ratio - %d" % i]
    S["f%d" % fail] = df_copy["f%d" % fail]
    S["f%d" % fail] = df_copy["f%d" % fail]
    S["id"] = df_copy["id"]
    S['tbtf'] = df_copy['tbtf']
    S = shuffle(S)
    S_test = pd.DataFrame(columns=S.columns)
    S_train = pd.DataFrame(columns=S.columns)
    fail_train = math.ceil((S.sum().loc["f%d" % fail]) * 0.9)  # nb de fail a mettre dans le train test

    S.to_excel("SSSd.xlsx")
    S = pd.read_excel('SSSd.xlsx')
    S.drop(['Unnamed: 0'],axis=1,inplace=True)
    for i in S.index:
        if i== 778:
            print(S.loc[i,"f%d" % fail])
        if S.loc[i,"f%d" % fail] == 1:
            if S_train.sum().loc["f%d" % fail] < fail_train:
                S_train = S_train.append(S.loc[i])
            else:
                S_test = S_test.append((S.loc[i]))
        if S.loc[i, 'tbtf'] == 1:
            if S_train.sum().loc["tbtf"] < 4:
                S_train = S_train.append(S.loc[i])
            else:
                S_test = S_test.append((S.loc[i]))
        if S.loc[i, 'tbtf'] != 1 and S.loc[i, 'f%d' % fail] != 1:
            if len(S_train) < math.ceil(0.7 * set_size):
                S_train = S_train.append((S.loc[i]))
            else:
                S_test = S_test.append((S.loc[i]))
    print(S_train)
    S_train['biais'] = 1
    S_test['biais'] = 1
    S['biais']=1
    S_test.drop(['id', 'tbtf'], axis=1, inplace=True)
    S_train.drop(['id', 'tbtf'], axis=1, inplace=True)
    S.drop(['id', 'tbtf'], axis=1, inplace=True)
    y_train = pd.DataFrame()
    X_train = pd.DataFrame()
    y_test = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = S_train["f%d" % fail]
    X_train = S_train.drop(columns='f%d' % fail)
    y_test = S_test["f%d" % fail]
    X_test = S_test.drop(columns='f%d' % fail)
    y_train = y_train.astype(('int64'))
    y_test = y_test.astype(('int64'))
    # --------------------------------------------Affichage des statistiques--------------------------------------------
    print('taille du train set')
    print(len(S_train))
    print('nombre de banques fesant faillites')
    print(S_train.sum().loc["f%d" % fail])
    # print(S_train.sum().loc["tbtf"])
    print('taille du test set')
    print(len(S_test))
    print('nombre de banques fesant faillites')
    print(S_test.sum().loc["f%d" % fail])
    print(X_train)
    return X_train,X_test,y_train,y_test


