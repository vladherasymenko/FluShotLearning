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
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
def warn(*args, **kwargs):
    pass
pd.set_option("display.max_columns", 100)

DATA_PATH = Path.cwd() / "data" 

def load_data(DATA_PATH):
    features_df = pd.read_csv(
        DATA_PATH / "training_set_features.csv", 
        index_col="respondent_id"
    )
    labels_df = pd.read_csv(
        DATA_PATH / "training_set_labels.csv", 
        index_col="respondent_id"
    )
    return features_df, labels_df
features_df, label_df = load_data(DATA_PATH)


X_train, X_eval, y_train, y_eval = train_test_split(features_df, label_df, test_size = 0.2, random_state = 42)
y_train_h1n1 = y_train["h1n1_vaccine"]
y_train_sf = y_train["seasonal_vaccine"]
y_eval_h1n1 = y_eval["h1n1_vaccine"]
y_eval_sf = y_eval["seasonal_vaccine"]

num_pipeline = Pipeline([('std_scaler', StandardScaler()),('imputer', SimpleImputer(strategy = "median"))])
cat_pipeline = Pipeline([('Encoder',OneHotEncoder())])
num_attribs = features_df.columns[features_df.dtypes != "object"].values
cat_attribs = features_df.columns[features_df.dtypes == "object"].values

preprocessor = ColumnTransformer([("num", num_pipeline, num_attribs)], remainder = "drop")
X_train_prepared = preprocessor.fit_transform(X_train)
X = preprocessor.transform(features_df)
y = label_df

warnings.warn = warn
warnings.filterwarnings("ignore", category=UserWarning)
X_eval_clean = preprocessor.transform(X_eval)
features_df, label_df = load_data(DATA_PATH)
regressor = MultiOutputRegressor(HistGradientBoostingRegressor(scoring="roc_auc",
                                                               max_iter=100,
                                                               l2_regularization=2000,
                                                               max_leaf_nodes=105,
                                                               learning_rate=0.1,
                                                               min_samples_leaf=1))

# training
regressor.fit(X_train_prepared, y_train)
# prediction
Y_predict_test = regressor.predict(X_eval_clean)
Y_predict_test[Y_predict_test < 0] = 0  # y є [0, 1]
Y_predict_test[Y_predict_test > 1] = 1
prec_test = roc_auc_score(y_eval, Y_predict_test)


Y_predict_train = regressor.predict(X_train_prepared)
Y_predict_train[Y_predict_train < 0] = 0  # y є [0, 1]
Y_predict_train[Y_predict_train > 1] = 1
prec2 = roc_auc_score(y_train, Y_predict_train)

Y_predict_tous = regressor.predict(X)
Y_predict_tous[Y_predict_tous < 0] = 0  # y є [0, 1]
Y_predict_tous[Y_predict_tous > 1] = 1
resp_id = pd.read_csv("test_set_features.csv")['respondent_id']
Y_predict = regressor.predict(X_eval_clean)
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
class DonneesTestCase(unittest.TestCase):
    # Le nombre de données à l'entrée doit être égale à celui à la sortie
    def test_data_quantity(self):
        self.assertEqual(self.y.shape, self.prediction.shape)
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
    # Vérifier s'il y a des valeurs NaNs
    def test_NaN(self):
        self.assertFalse(np.any(np.isnan(self.X)))
        self.assertFalse(np.any(np.isnan(self.y)))
        self.assertFalse(np.any(np.isnan(self.prediction)))
    # Vérifier s'il y a des valeurs infinies
    def test_inf(self):
        self.assertFalse(np.any(np.isinf(self.X)))
unittest.main()