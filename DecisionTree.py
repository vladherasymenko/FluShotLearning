import pandas as pd
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
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
X_train = pd.get_dummies(X_train)
X_train = X_train.fillna(X_train.mean())

#Validation set
X_test = np.array(X_train.iloc[25200:])
y_test = np.array(y_train.iloc[25200:])

#Train set
X_train = np.array(X_train.iloc[:25200])
y_train = np.array(y_train.iloc[:25200])


#Normalization
sc = MinMaxScaler()
X_train_norm = sc.fit_transform(X_train)
X_test_norm = sc.transform(X_test)


regressor = RandomForestRegressor(n_estimators=128, criterion="squared_error", n_jobs=4)
regressor.fit(X_train, y_train)
Y_predict = regressor.predict(X_test)
print(roc_auc_score(y_test, Y_predict))