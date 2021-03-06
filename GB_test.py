#désactiver les notification de sklearn
def warn(*args, **kwargs):
    pass

import unittest
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import warnings

warnings.warn = warn
warnings.filterwarnings("ignore", category=UserWarning)

# sauvegarder le modèle
def save_mode(X_to_predict, regressor):
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

    print("Le modèle a été sauvegardé")
    return True

# téléchargwer les données
X = (pd.read_csv("restored_train.csv"))
y = pd.read_csv("training_set_labels.csv").drop(['respondent_id'], axis=1)

X_to_predict = (pd.read_csv("restored_test.csv"))

#PCA - n'aide pas du tout
#pca = PCA(103)
#X = pca.fit_transform(X)
#X_to_predict = pca.fit_transform(X_to_predict)

resp_id = pd.read_csv("test_set_features.csv")['respondent_id']

# model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1488)
regressor = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100,
                                                           max_depth=5,
                                                           learning_rate=0.1,
                                                           max_features=95,
                                                           criterion="mse",
                                                           max_leaf_nodes=105))

# entraînement
regressor.fit(X_train, y_train)
# prediction
Y_predict_test = regressor.predict(X_test)
Y_predict_test[Y_predict_test < 0] = 0  # y є [0, 1]
Y_predict_test[Y_predict_test > 1] = 1
prec_test = roc_auc_score(y_test, Y_predict_test)

print("Précision test :", prec_test)

Y_predict_train = regressor.predict(X_train)
Y_predict_train[Y_predict_train < 0] = 0  # y є [0, 1]
Y_predict_train[Y_predict_train > 1] = 1
prec2 = roc_auc_score(y_train, Y_predict_train)
print("Précision train :", prec2)

save_mode(X_to_predict, regressor)