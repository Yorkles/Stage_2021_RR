#--------------------------------------------Imports--------------------------------------------------------------------
import pandas as pd
import random
import sys
import create_series
import create_sets
import run_regression
import create_ratios
#--------------------------------------------Initialisation-------------------------------------------------------------
create_ratios()
best_f1_score=0
best_recall_score=0
best_ratio=0
best_size=0
best_seed=0
df=pd.DataFrame()
best_res=pd.DataFrame(columns=['fail','ratio','size','seed','f1','recall'])
b=pd.DataFrame(columns=best_res.columns)

#---------------------------------------Recherche des meilleurs resultats-----------------------------------------------
for fail in range(1,4):
    best_f1_score=0
    for size in range(3, 7):
        for ratio in range(1, 4):
            for seed in range(1, 10):
                df = create_series.create_series(size, ratio)
                X_train, X_test, y_train, y_test = create_sets.create_sets(size, df, 0.5, fail, seed)
                f1_score, recall_score,model_cv = run_regression.run_reg(X_train, y_train, X_test, y_test)
                if f1_score >= best_f1_score:
                    best_f1_score = f1_score
                    best_size = size
                    best_ratio = ratio
                    best_seed = seed
                    best_recall_score=recall_score
    print(best_f1_score)
    print(best_seed)
    print(best_ratio)
    print(best_size)
    new_row = {'fail': fail, 'ratio': best_ratio, 'size': best_size, 'seed': best_seed, 'f1': best_f1_score,
                   'recall': best_recall_score}
    b.loc[0, 'fail'] = fail
    b.loc[0, 'ratio'] = best_ratio
    b.loc[0, 'size'] = best_size
    b.loc[0, 'seed'] = best_seed
    b.loc[0, 'f1'] = best_f1_score
    b.loc[0, 'recall'] = best_recall_score

    best_res = best_res.append(b.loc[0])
    best_res.to_excel("best_res.xlsx" % fail)


