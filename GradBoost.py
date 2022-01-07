#TODO tests + validation
# [0,1] pas de NaN, inf...
# Tests de perf. >0.65 - acceptable, >0.8 - correct, >0.85 - excellent
# nombre de données
# One-hot - nombre de colonnes

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


#download data
X = pd.read_csv("restored_train.csv")
y = pd.read_csv("training_set_labels.csv").drop(['respondent_id'], axis=1)
X_to_predict = pd.read_csv("test_set_features.csv")

#PCA
#pca = PCA(105)
#X = pca.fit_transform(X)

#Séparer ID et le reste du jeux de données
resp_id = X_to_predict['respondent_id']
X_to_predict = X_to_predict.drop(['respondent_id'], axis=1)
X_to_predict = pd.get_dummies(X_to_predict)
X_to_predict = X_to_predict.fillna(X_to_predict.mean())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)
regressor = MultiOutputRegressor(HistGradientBoostingRegressor(scoring="roc_auc",
                                                               max_iter=50,
                                                               l2_regularization=100,
                                                               max_leaf_nodes=105,
                                                               learning_rate=0.1,
                                                               min_samples_leaf=1))

regressor.fit(X_train, y_train)
Y_predict_test = regressor.predict(X_test)
Y_predict_test[Y_predict_test < 0] = 0  # y є [0, 1]
Y_predict_test[Y_predict_test > 1] = 1
prec = roc_auc_score(y_test, Y_predict_test)

print("Précision test :", prec)

Y_predict_train = regressor.predict(X_train)
Y_predict_train[Y_predict_train < 0] = 0  # y є [0, 1]
Y_predict_train[Y_predict_train > 1] = 1
prec2 = roc_auc_score(y_train, Y_predict_train)
print("Précision train :", prec2)


Y_predict = regressor.predict(X_to_predict)
# y є [0, 1]
Y_predict[Y_predict < 0] = 0
Y_predict[Y_predict > 1] = 1
result = pd.DataFrame(Y_predict)
result["respondent_id"] = resp_id
result = result.rename(columns={0: "h1n1_vaccine", 1: "seasonal_vaccine"})
cols = ["respondent_id", "h1n1_vaccine", "seasonal_vaccine"]
result = result[cols]
result = result.astype({"respondent_id": int})
result.to_csv("model.csv", index=False)

#0.325 L2
#200 iter
#max_leaf_nodes 67
#LR 0.06
#46 ou 40

"""
,
                                                               max_iter=200,
                                                               learning_rate=0.0666,
                                                               l2_regularization=150,
                                                               max_leaf_nodes=15,
                                                               min_samples_leaf=50
"""