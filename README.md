# FluShotLearning

GradBoost.py - l'implémentation de l'algorithme Gradient Boosting (le modèle principal pour l'instant). La précision maximale atteinte est de 85.95% (sur DrivenData) et de ~87.5% sur la machine locale. 

DecisionTree.py - la forêt aléatoire. Les résultats ne sont pas satisfaisants. 

ReseauNeuronal.py - le réseau de neurones. Les résultats ne sont pas satisfaisants, non plus.


restored_train.csv - le jeu de données d'entraînement où les valeurs "NaN" ont été remplacées par des prédictions faites à partir des données existantes (cf. Sklearn -> Imputer, par exemple) 

restored_test.csv - le jeu de données de test où les valeurs "NaN" ont été remplacées par des prédictions faites à partir des données existantes (cf. Sklearn -> Imputer, par exemple) 

unit_test.py - les tests unitaires (pour Gradient Boosting) -> tous les tests ont passés la vérification avec succès

courbes.zip - contient les courbes qui montrent la dépendence entre la précision (sur les jeux de données test/train) et les paramètres du modèle (Gradient Boosting pour le moment)   
