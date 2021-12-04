import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#download data
X_train = pd.read_csv("training_set_features.csv").drop(['respondent_id'], axis=1)
y_train = pd.read_csv("training_set_labels.csv").drop(['respondent_id'], axis=1)

for i in range(15):  # ! A optimiser
    X_train.iloc[:, [i]] = X_train.iloc[:, [i]].fillna(X_train.iloc[:, [i]].mean())

for i in range(21,31):  # ! A optimiser
    X_train.iloc[:, [i]] = X_train.iloc[:, [i]].fillna(X_train.iloc[:, [i]].mean())

for i in range(33,len(X_train.columns)):  # ! A optimiser
    X_train.iloc[:, [i]] = X_train.iloc[:, [i]].fillna(X_train.iloc[:, [i]].mean())


#pre-processing
categorical = X_train.columns[-14:]
X_train = pd.get_dummies(X_train)
#X_train = X_train.fillna(X_train.mean())

#Validation set
X_test = np.array(X_train.iloc[25350:])
y_test = np.array(y_train.iloc[25350:])

#Train set
X_train = np.array(X_train.iloc[:25350])
y_train = np.array(y_train.iloc[:25350])

#Normalization
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)


# TODO Essayer d'utiliser le support des "categorical features"
regressor = MultiOutputRegressor(HistGradientBoostingRegressor(loss="squared_error",
                                                                     scoring="roc_auc",
                                                                     max_iter=4000,
                                                                     learning_rate=0.03,
                                                                     l2_regularization=0.001))
max_prec = 0
for i in range(20):
    regressor.fit(X_train, y_train)
    Y_predict = regressor.predict(X_test)
    prec = roc_auc_score(y_test, Y_predict)
    print("Essai №", i+1, "/", "20; precision =", prec)
    if prec > max_prec:
        max_prec = prec

print(round(max_prec*100, 3), "%")