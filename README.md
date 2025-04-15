# Analyse de Sentiment des Critiques IMDB

Ce projet réalise une analyse de sentiment sur un dataset de critiques de films IMDB, en utilisant différentes techniques de traitement du langage naturel (NLP) et d'apprentissage automatique.

## Structure du Projet

```
.
├── IMDB Dataset.csv       # Dataset original des critiques IMDB
├── main.py                # Script principal d'analyse
├── preprocessed_imdb.csv  # Dataset prétraité
├── ML/                    # Résultats des modèles de machine learning
│   ├── ini/               # Résultats des modèles initiaux
│   └── improved/          # Résultats des modèles améliorés
└── visualisation/         # Visualisations générées
    ├── raw/               # Visualisations des données brutes
    └── preprocessed/      # Visualisations des données prétraitées
```

## Fonctionnalités

1. **Prétraitement des données**
   - Suppression des balises HTML
   - Suppression des caractères non alphabétiques
   - Conversion en minuscules
   - Tokenisation
   - Suppression des stop words
   - Stemming

2. **Visualisation des données**
   - Distribution des sentiments
   - Analyse de la longueur des critiques
   - Nuages de mots pour les critiques positives et négatives
   - Comparaison avant/après prétraitement
   - Mots les plus fréquents par sentiment

3. **Modèles de Machine Learning**
   - Modèles initiaux (Bag of Words)
     - Naive Bayes
     - Régression Logistique
     - Random Forest
     - SVM Linéaire
   - Modèles améliorés (TF-IDF)
     - Versions optimisées des modèles ci-dessus

## Résultats

Les modèles ont été évalués sur leur précision à classifier correctement les critiques comme positives ou négatives. Les matrices de confusion et les rapports de classification sont disponibles dans les dossiers ML/ini et ML/improved.

Le meilleur modèle obtient une précision supérieure à 85% sur l'ensemble de test.

## Prérequis

Pour exécuter ce projet, vous aurez besoin des bibliothèques Python suivantes :

- pandas
- numpy
- matplotlib
- seaborn
- wordcloud
- nltk
- scikit-learn

## Utilisation

1. Clonez ce dépôt
2. Installez les dépendances : `pip install -r requirements.txt`
3. Exécutez le script principal : `python main.py`

## Licence

Ce projet est sous licence MIT.