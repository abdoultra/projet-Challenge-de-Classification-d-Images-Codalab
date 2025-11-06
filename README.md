Rapport du Challenge de Classification d’Images Codalab 2025
Abdoulaye Traoré
Score Codalab : 0.7603
1. Méthodes Testées et Résultats Obtenus
1.1. Méthode des K plus proches voisins (KNN)
• Séparation des données : 80% pour l'entraînement / 20% pour la validation
• Optimisation de K : Meilleur K trouvé à 5 après tests
• Résultats :
o Précision en validation : 60%
o Score Codalab : 0.48
• Limites :
o Algorithme lent sur dataset volumineux (images haute dimension)
o Faible efficacité sur ce challenge
1.2. Régression Logistique Multivariée
• Séparation des données : 70% pour l'entraînement / 30% pour le test
• Hyperparamètres :
o Learning rate : 0.01
o Nombre d'epochs : 100000
• Résultats :
o Précision en entraînement : 80%
o Précision en test : 77%
• Commentaires :
o Méthode simple, mais limitée pour traiter des images complexes
1.3. Réseau de Neurones à Couches Linéaires (MLP)
• Architecture :
o 2 couches linéaires avec activation ReLU
• Paramètres :
o Learning rate : 0.001
o Nombre d'epochs : 10000
• Résultats :
o Précision en entraînement : 99%
o Précision en test : 93%
• Limites :
o Surapprentissage constaté sur les données d'entraînement
1.4. Réseau de Neurones Convolutifs (CNN)
• Architecture :
o 3 couches convolutives avec BatchNorm, Pooling, Dropout
o 2 couches fully-connected (FC)
• Paramètres d'entraînement :
o Learning rate : 0.0005
o Scheduler : Cosine Annealing
o Early Stopping : patience 7 epochs
o Nombre d'epochs : 50
o Batch size : 64
• Data Augmentation :
o RandomHorizontalFlip
o RandomRotation
o RandomResizedCrop
• Résultats :
o Précision sur validation : 76.15%
o Score Codalab : 0.7603
• Classement : 6e sur le leaderboard
2. Conclusion et Perspectives
• CNN performant grâce à :
o Profondeur et complexité contrôlée du modèle
o Régularisation efficace (dropout, batchnorm)
o Data augmentation pertinente
• Pistes d'amélioration :
o Tester des architectures plus profondes (ResNet, DenseNet)
o Optimisation des hyperparamètres (GridSearch, Bayesian Optimization)
o Augmenter la Data Augmentation (Cutout, Mixup)
3. Format de Soumission
• Fichier CSV : images_test_predictions.csv
• Format exigé par Codalab :
• Généré via numpy.savetxt pour correspondre aux spécifications du script evaluate.py de 
Codalab.
4. Résultats sur Codalab
Méthode Score
K plus proches voisins 0.48
Régression Logistique 0.52
Réseau Neuronal Linéaire 0.55
Réseau CNN 0.7603
5. Remarques Finales
Le CNN proposé a permis d’obtenir un score correct dans le challenge, mais il reste encore une marge 
de progression pour atteindre les scores les plus élevés (>0.80).
L’expérimentation avec des architectures plus complexes et des stratégies de régularisation avancées 
est envisagée pour améliorer encore la performance.
