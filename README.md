# FluShotLearning

GradBoost.py - l'implémentation de l'algorithme Gradient Boosting. La précision maximale atteinte est de 86.17% (sur DrivenData) et de ~91.2% sur la machine locale. 

GB_Boosting.py (le modèle principal pour l'instant) - la combinaison de technique de Bagging avec le Gradient Boosting. Les résultats sont légèrement mieux (que sans Bagging). Entrée utilisateur est aussi disponible à travers de la fontion predict_one(). 

DecisionTree.py - la forêt aléatoire. Les résultats ne sont pas satisfaisants. 

ReseauNeuronal.py - le réseau de neurones. Les résultats ne sont pas satisfaisants, non plus.


restored_train.csv - le jeu de données d'entraînement où les valeurs "NaN" ont été remplacées par des prédictions faites à partir des données existantes (cf. Sklearn -> Imputer, par exemple) 

restored_test.csv - le jeu de données de test où les valeurs "NaN" ont été remplacées par des prédictions faites à partir des données existantes (cf. Sklearn -> Imputer, par exemple) 

unit_test.py - les tests unitaires (pour Gradient Boosting) -> tous les tests ont passés la vérification avec succès

courbes.zip - contient les courbes qui montrent la dépendence entre la précision (sur les jeux de données test/train) et les paramètres du modèle (Gradient Boosting pour le moment)   
