# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import openpyxl
import pandas as pd
import datetime as dt
import numpy as np
#import matplotlib.pyplot as plt


# ARRANGE THE DATA
def datas(df):

    df['to_drop'] = 0
    for i in df.index:
        if df.loc[i, 'referenceyear'] < 1930 and (df.loc[i, 'f1'] == 1 or df.loc[i, 'f2'] == 1 or df.loc[i, 'f3'] == 1):
            df.loc[df['id'] == df.loc[i, 'id'], 'to_drop'] = 1
    df = df.drop(df[df.to_drop == 1].index)
    print(df.id.nunique())
    df['i1'] = 0

    for i in df.index:
        if df.loc[i, 'f1'] == 1:
            df.loc[df['id'] == df.loc[i, 'id'], 'i1'] = 1
    for i in df.index:
        if df.loc[i, 'referenceyear'] != 1929:
            df = df.drop(i)
    return df


def create_inputs(df):
    df_clean = df[['i1']]
    df_clean['cap_ratio'] = (df['capital'] + df['capital_surplus']) / df['tot_assets']
    df_clean['liq_ratio'] = df['cash'] / df['tot_assets']

    df_clean['i1'] = df['i1']
    np_clean = df_clean.to_numpy()
    #df_clean.to_excel("clean.xlsx")
    return np_clean


# activation function
# gradients
# --------------------------------------------Activation function--------------------------------------------------------
def activation(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------Dérivée de l 'activation ------------------------------------------------
def der_act(z):
    return activation(z) * (1 - activation(z))


# ---------------------------------------------Usual gradient-----------------------------------------------------------
def backward(w, learning_rate, X_train, Y_train, y, z):
    w1 = 0
    for k in range(0, w.shape[0]):
        for i in range(0, X_train.shape[0]):
            w1 = w1 + ((y[i] - Y_train[i]) * der_act(z[i]) * X_train[i, k])
        w[k] = w[k] - (learning_rate * w1 / X_train.shape[0])
        w1 = 0


    return w


# --------------------------------DECISION BOUNDARY----------------------------------------------------------------------
def dec_bound(y):
    for i in range(y.shape[0]):
        if y[i] > 0.5:
            y[i] = 1
        if y[i] < 0.5:
            y[i] = 0

    return y


def log_reg(X_train, Y_train):
    W = np.random.rand(X_train.shape[1], 1)
    Z = np.zeros(Y_train.shape)
    learning_rate = 0.1
    y = np.zeros(Y_train.shape)
    cost=0
    print(X_train.shape)
    cost_hist=np.zeros(1)
    for i in range(100):
        Z = np.dot(X_train, W)
        y = activation(Z)
        y = dec_bound(y)
        W = backward(W, learning_rate, X_train, Y_train, y, Z)
        cost = np.mean((y - Y_train) ** 2)
        cost_hist=np.append(cost_hist,cost)

    return W,cost_hist


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_excel('Datas.xlsx', index_col=None)
    df = datas(df)
    set = create_inputs((df))
    idx = np.random.permutation(set.shape[0])
    train_idx, test_idx = idx[:int(round(0.7 * set.shape[0]))], idx[int((0.7 * set.shape[0])):]
    train, test = set[train_idx, :], set[test_idx, :]
    print(train.shape)
    print(test.shape)
    X_train = train[:, 1:]
    Y_train = train[:, 0:1]
    X_test = test[:, 1:]
    Y_test = test[:, 0:1]
    print(X_train.shape)
    print(Y_train.shape)
    W,cost_hist = log_reg(X_train, Y_train)
    Z = np.dot(X_test, W)
    y = activation(Z)
    y = dec_bound(y)
    SSE = np.mean((y - Y_test) ** 2)
    print(SSE)
    print(W)
    #print(cost_hist)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# clean of the datatset y a des trucs dans le df qui srveent a rien
# tout convert en int pasque ca casse les couiles sionon tnat qu on garde l id on pourra retrouver les nom
# faire l fractionspreparer el fwd balancer le back et hop on a fini ce soir ya juste a joiuer avec les variable te faire es pyplot j y passe al nuit si il faut
