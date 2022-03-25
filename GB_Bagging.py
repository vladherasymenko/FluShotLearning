#désactiver les notification de sklearn
def warn(*args, **kwargs):
    pass

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingRegressor
import time
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


def predict_one(data, regressor):
    prediction = regressor.predict(data)
    print("Le probabilité de se faire vacciner contre H1N1 est ~",np.round(prediction[0][0]*100,2),"%")
    print("Le probabilité de se faire vacciner contre la grippe saisonière est ~",np.round(prediction[0][1]*100,2),"%")

start_time = time.time()
# télécharger les données
X = (pd.read_csv("restored_train.csv"))
y = pd.read_csv("training_set_labels.csv").drop(['respondent_id'], axis=1)

X_to_predict = (pd.read_csv("restored_test.csv"))

#PCA - n'aide pas du tout
#pca = PCA(103)
#X = pca.fit_transform(X)
#X_to_predict = pca.fit_transform(X_to_predict)

resp_id = pd.read_csv("test_set_features.csv")['respondent_id']

# modèle "faible"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1488)
regressor = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100,
                                                           max_depth=5,
                                                           learning_rate=0.1,
                                                           max_features=105,
                                                           criterion="squared_error",
                                                           max_leaf_nodes=105,
                                                           loss="squared_error"))

# entraînement
regr_bagging = BaggingRegressor(base_estimator=regressor, n_estimators=2,
                                random_state=0, n_jobs=8, max_samples=1.0).fit(X_train, y_train)


#regressor.fit(X_train, y_train)
# prediction
print("Pour la preimière personne dans le jeu de test : ")
predict_one(np.array(X_test.iloc[0]).reshape(1, -1), regr_bagging)

Y_predict_test = regr_bagging.predict(X_test)
Y_predict_test[Y_predict_test < 0] = 0  # y є [0, 1]
Y_predict_test[Y_predict_test > 1] = 1
prec_test = roc_auc_score(y_test, Y_predict_test)

print("\n\nPrécision test :", prec_test)

Y_predict_train = regr_bagging.predict(X_train)
Y_predict_train[Y_predict_train < 0] = 0  # y є [0, 1]
Y_predict_train[Y_predict_train > 1] = 1
prec2 = roc_auc_score(y_train, Y_predict_train)
print("Précision train :", prec2)

save_mode(X_to_predict, regr_bagging)
print("--- %s secondes ---" % (time.time() - start_time))