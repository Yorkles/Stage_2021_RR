#---------------------------Imports-------------------------------------------------------------------------------------
import create_ratios
import pandas as pd
import random
import matplotlib.pyplot as plt
import create_series
import create_sets
import sklearn.metrics as mt
import run_regression
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression

#----------------------------Affichage des resultats ------------------------------------------------------------------
best= pd.read_excel('best_res.xlsx', index_col=None)
fail=best.loc[0,'fail']
ratio=best.loc[0,'ratio']
size=best.loc[0,'size']
seed=best.loc[0,'seed']
df = create_series.create_series(size, ratio)
X_train, X_test, y_train, y_test = create_sets.create_sets(size, df, 0.5, fail, seed)
f1_score, recall_score, model_cv = run_regression.run_reg(X_train, y_train, X_test, y_test)
model = LogisticRegression(max_iter=300, class_weight='balanced', penalty='l2', C=0.1,solver='liblinear',verbose=1)

fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')
xs = X_test['cap_ratio - 1'].tolist()
zs = model_cv.predict_proba(X_test)[:,1].tolist()
ys = X_test['liq_ratio - 1'].tolist()
ax.scatter(xs, ys, zs)
zs = model_cv.predict(X_test).tolist()
ax.scatter(xs, ys, zs,'r')
ax.set_xlabel('Ratio de capital')
ax.set_ylabel('Ratio de liquidité')
ax.set_zlabel('Faillite')

fig=plt.figure(2)
y_pred = model_cv.predict(X_test)
cf=mt.confusion_matrix(y_test,y_pred)
sns.heatmap(cf,annot = True)


df_copy = pd.read_excel('datas_series.xlsx', index_col=None)

id=0
df_copy['biais']=1
for i in df_copy.index:
    if df_copy.loc[i,'f1']==1:
        id=df_copy.loc[i,'id']
        df_copy.loc[i,'id']=id*2000
        df.drop(df[df.id == id].index, inplace=True)

X=df_copy[['cap_ratio - 1','liq_ratio - 1','cap_ratio - 2','liq_ratio - 2','cap_ratio - 3','liq_ratio - 3','biais']]
y=df_copy[['f1']]
y_pred = model_cv.predict(X)
f1_score = mt.f1_score(y, y_pred)
recall_score=mt.recall_score(y, y_pred)
confusion_matrix=mt.confusion_matrix(y, y_pred)
print('recall score =', mt.recall_score(y, y_pred))
print('f1 score =', mt.f1_score(y, y_pred))
print(mt.confusion_matrix(y, y_pred))
plt.show()

