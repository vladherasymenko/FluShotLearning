import tkinter
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from joblib import load
from pathlib import Path
import numpy as np
import pandas as pd
import time
import os

name = ""
global text

def select_file():
    global text
    filetypes = (
        ('text files', '*.csv'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Choisir un fichier',
        initialdir='',
        filetypes=filetypes)

    cut_name = filename[filename.rfind('/')+1:]
    text.set(f"Fichier courant : {cut_name}")
    
    global name
    name = filename
    return

def prediction():
    global name
    if(name == ""):
        showinfo(title='Fichier', message='Veuillez sélectionner un fichier')
        return
    DATA_PATH = Path.cwd()
    features_df = None
    while features_df is None:
        try:
            file_name = name
            features_df = pd.read_csv(
            DATA_PATH / file_name,
            index_col="respondent_id")
        except:
            print("Fichier non valide")

    system_name = 'gradient_boosting_final_5'
    system_load = system_name + '.joblib'

    final_system = load(os.getcwd()+"\\"+system_load)

    predictions = final_system.predict_proba(features_df)

    y_preds = pd.DataFrame(
           {
               "h1n1_vaccine": predictions[0][:, 1],
               "seasonal_vaccine": predictions[1][:, 1],
           }, index = features_df.index)
    pred_save = 'predictions'+ '.csv'
    y_preds.to_csv(pred_save)
    tkinter.messagebox.showinfo(title = 'Succès', message = "Le fichier de prédiction '" + pred_save + "' a été créé")
    return


def exit_window():
    root.destroy()
    raise SystemExit

root = tkinter.Tk()

root.title('FluShotLearning')
root.geometry('410x410')

label_start = tkinter.Label(root, text = "Entrée utilisateur",font=("Arial Black", 13))
label_start.grid(row = 0, column = 0, columnspan = 2,padx= 70,pady=15)

text = tkinter.StringVar()
text.set("Fichier courant : non sélectionné")

label_fichier = tkinter.Label(root, textvariable = text,font=("Times New Roman", 14))
label_fichier.grid(row = 1, column = 0, columnspan = 2,padx= 70,pady=15)

but_plot = tkinter.Button(root, text = "Ouvrir le fichier" , command = lambda : select_file() , font=("Times New Roman", 16))
but_plot.grid(row= 2, column = 0, columnspan =2, pady =20)

but_plot = tkinter.Button(root, text = "Lancer" , command = lambda : prediction() , font=("Times New Roman", 16))
but_plot.grid(row= 3, column = 0, columnspan =2, pady =20)

button_close = tkinter.Button(root, text = 'Quitter',command = exit_window, bg = 'red', fg = 'white',font=("Arial Black", 12))
button_close.grid(row= 100, column = 0, columnspan = 2, pady =20)

root.mainloop()
