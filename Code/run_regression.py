#-----------------------------------------Imports-----------------------------------------------------------------------
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

#------------------------------------------------Execution de la regression logistique---------------------------------
def run_reg(X_train,y_train,X_test,y_test):
    # Grid search
    grid = {"C": np.logspace(-3, 3, 7), "solver": ["liblinear"]}  # l1 lasso l2 ridge
    model = LogisticRegression(max_iter=300, class_weight='balanced', penalty='l2')
    model_cv = GridSearchCV(model, grid, cv=10, scoring='f1')
    model_cv.fit(X_train, y_train)
    y_pred = model_cv.predict(X_test)
    y_probs = model_cv.predict_proba(X_test)


    f1_score = mt.f1_score(y_test, y_pred)
    recall_score=mt.recall_score(y_test, y_pred)
    confusion_matrix=mt.confusion_matrix(y_test, y_pred)
    print('recall score =', mt.recall_score(y_test, y_pred))
    print('f1 score =', mt.f1_score(y_test, y_pred))
    print(mt.confusion_matrix(y_test, y_pred))

    coeff = pd.DataFrame()
    coeff['Feature'] = X_train.columns
    coeff['Coefficient Estimate'] = pd.Series(model_cv.best_estimator_.coef_[0])
    print (coeff)
    print("tuned hpyerparameters :(best parameters) ", model_cv.best_params_)



    return f1_score,recall_score,model_cv

