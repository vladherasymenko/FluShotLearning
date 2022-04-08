import tkinter
from joblib import load
from pathlib import Path
import numpy as np
import pandas as pd
import time

def test():
    DATA_PATH = Path.cwd()
    features_df = None
    while features_df is None:
        try:
            file_name = 'test_set_features.csv'
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



def exit_window():
    """
    Функція для виходу з програми
    """
    root.destroy()
    raise SystemExit

root = tkinter.Tk()

root.title('FluShotLearning')
root.geometry('440x360')

label_start = tkinter.Label(root, text = "Entrée utilisateur",font=("Arial Black", 9))
label_start.grid(row = 0, column = 0, columnspan = 2,padx= 15,pady=15)


but_plot = tkinter.Button(root, text = "Lancer" , command =lambda: Pauel([10.5,10.5]) , font=("Times New Roman", 12))
but_plot.grid(row= 1, column = 0, columnspan =2, pady =20)

button_close = tkinter.Button(root, text = 'Quitter',command = exit_window, bg = 'red', fg = 'white',font=("Arial Black", 8))
button_close.grid(row= 100, column = 0, columnspan = 2, pady =20)
