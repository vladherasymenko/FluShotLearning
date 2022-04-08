from joblib import load
from pathlib import Path
import numpy as np
import pandas as pd
import time
DATA_PATH = Path.cwd()
features_df = None
while features_df is None:
    try:
        file_name = input("Entrer le fichier contenant les donnees: ")
        features_df = pd.read_csv(
        DATA_PATH / file_name,
        index_col="respondent_id")
    except:
        print("Fichier non valide")

system_name = 'gradient_boosting_final_5'
system_load = system_name + '.joblib'
final_system = load(DATA_PATH /system_load)

predictions = final_system.predict_proba(features_df)
y_preds = pd.DataFrame(
       {
           "h1n1_vaccine": predictions[0][:, 1],
           "seasonal_vaccine": predictions[1][:, 1],
       }, index = features_df.index)
pred_save = 'predictions_' + system_name + '.csv'
y_preds.to_csv(DATA_PATH/pred_save)
print("Le fichier de prediction " + pred_save + " a ete cree")
