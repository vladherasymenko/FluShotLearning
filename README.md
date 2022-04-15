# FluShotLearning

GradBoost.py (le modèle principal) - l'implémentation de l'algorithme Gradient Boosting. La précision maximale atteinte est de 86.17% (sur DrivenData) et de ~91% sur la machine locale. 

GB_Boosting.py - la combinaison de technique de Bagging avec le Gradient Boosting. Les résultats sont légèrement mieux (que sans Bagging). 

Entrée utilisateur avec l'enterface graphique est disponible à travers du fichier GUI. Pour l'utiliser il faut mettre GUI.py et fichier .joblib dans le même répertoire et lancer ce programme depuis terminal (python GUI.py). Il faut installer Python et tous les prérequis préalablement. 

DecisionTree.py - la forêt aléatoire. Les résultats ne sont pas satisfaisants. 

ReseauNeuronal.py - le réseau de neurones. Les résultats ne sont pas satisfaisants, non plus.


restored_train.csv - le jeu de données d'entraînement où les valeurs "NaN" ont été remplacées par des prédictions faites à partir des données existantes (cf. Sklearn -> Imputer, par exemple) 

restored_test.csv - le jeu de données de test où les valeurs "NaN" ont été remplacées par des prédictions faites à partir des données existantes (cf. Sklearn -> Imputer, par exemple) 

unit_test.py - les tests unitaires (pour Gradient Boosting) -> tous les tests ont passé la vérification avec succès

courbes.zip - contient les courbes qui montrent la dépendence entre la précision (sur les jeux de données test/train) et les paramètres du modèle (Gradient Boosting pour le moment)   
