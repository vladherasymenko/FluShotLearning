def warn(*args, **kwargs):
    pass

import unittest
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import warnings

warnings.warn = warn
warnings.filterwarnings("ignore", category=UserWarning)

#download data
X = pd.read_csv("restored_train.csv")
y = pd.read_csv("training_set_labels.csv").drop(['respondent_id'], axis=1)

X_to_predict = pd.read_csv("restored_test.csv")

resp_id = pd.read_csv("test_set_features.csv")['respondent_id']

# model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=321)
regressor = MultiOutputRegressor(HistGradientBoostingRegressor(scoring="roc_auc",
                                                               max_iter=100,
                                                               l2_regularization=2000,
                                                               max_leaf_nodes=105,
                                                               learning_rate=0.1,
                                                               min_samples_leaf=1))

# training
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

Y_predict_tous = regressor.predict(X)
Y_predict_tous[Y_predict_tous < 0] = 0  # y є [0, 1]
Y_predict_tous[Y_predict_tous > 1] = 1

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

"""
class DonneesTestCase(unittest.TestCase):
    def setUp(self):
        self.X = np.array(X)
        self.y = np.array(y).transpose()
        self.prediction = np.array(Y_predict_tous).transpose()
        self.precision = np.array(prec_test)

    # Valeurs de y sont compris entre 0 et 1
    def test_y_in_0_1(self):
        self.assertTrue(np.all(np.logical_and(self.y[0] >= 0, 1 >= self.y[0])))
        self.assertTrue(np.all(np.logical_and(self.y[1] >= 0, 1 >= self.y[1])))

        self.assertTrue(np.all(np.logical_and(self.prediction[0] >= 0, 1 >= self.prediction[0])))
        self.assertTrue(np.all(np.logical_and(self.prediction[1] >= 0, 1 >= self.prediction[1])))

    # Le nombre de données à l'entrée doit être égale à celui à la sortie
    def test_data_quantity(self):
        self.assertEqual(self.y.shape, self.prediction.shape)

    # Vérifier s'il y a des valeurs NaNs
    def test_NaN(self):
        self.assertFalse(np.any(np.isnan(self.X)))
        self.assertFalse(np.any(np.isnan(self.y)))
        self.assertFalse(np.any(np.isnan(self.prediction)))

    # Vérifier s'il y a des valeurs infinies
    def test_inf(self):
        self.assertFalse(np.any(np.isinf(self.X)))

class PredictionTestCase(unittest.TestCase):
    def setUp(self):
        self.precision = np.array(prec_test)

    # La précision excellente est 85% ou mieux
    def test_precision_excelente(self):
        self.assertTrue(self.precision >= 0.85)

    # La précision correcte est 80% ou mieux
    def test_precision_correcte(self):
        self.assertTrue(self.precision >= 0.8)

    # La précision acceptable est 65% ou mieux
    def test_precision_acceptable(self):
        self.assertTrue(self.precision >= 0.65)
"""