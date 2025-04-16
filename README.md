# 🎬 Analyse de Sentiment sur les Critiques de Films IMDB

Ce projet a pour objectif de classifier automatiquement des critiques de films IMDB comme **positives** ou **négatives**, à l'aide de techniques de **traitement du langage naturel (NLP)** et de **machine learning**.

> 📊 Dataset utilisé : [IMDB Large Movie Review Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (Kaggle)

---

## 🗂️ Structure du Projet

```
.
├── IMDB Dataset.csv         # Dataset brut original
├── preprocessed_imdb.csv    # Dataset après nettoyage
├── main.py                  # Script principal du projet
├── ML/
│   ├── ini/                 # Résultats des modèles initiaux (Bag of Words)
│   └── improved/            # Résultats avec TF-IDF et optimisation
└── visualisation/
    ├── raw/                 # Graphiques sur les données brutes
    └── preprocessed/        # Graphiques après prétraitement
```

---

## ⚙️ Fonctionnalités

### 1. 📚 Prétraitement des Données
- Nettoyage HTML
- Suppression des caractères spéciaux
- Passage en minuscules
- Tokenisation
- Suppression des *stop words*
- Stemming

### 2. 📈 Visualisation
- Distribution des sentiments
- Longueur des critiques (avant/après)
- Nuages de mots
- Mots les plus fréquents

### 3. 🤖 Modélisation
**Modèles testés :**
- Naive Bayes
- Régression Logistique
- Random Forest
- SVM Linéaire

**Approches comparées :**
- Bag of Words
- TF-IDF (avec optimisation des hyperparamètres)

---

## 🏆 Résultats
Le meilleur modèle (Régression Logistique + TF-IDF) atteint une **précision de 88.79%** sur l'ensemble de test ✅  
Les résultats complets (rapports, matrices de confusion) sont dans les dossiers `ML/ini/` et `ML/improved/`.

---

## 📦 Prérequis
Installez les bibliothèques nécessaires via :

```bash
pip install -r requirements.txt
```

Modules requis :
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `wordcloud`
- `nltk`
- `scikit-learn`

---

## 🚀 Exécution

```bash
git clone https://github.com/Mouad-cmd-cpu/IMDB-Sentiment-Analysis.git
cd IMDB-Sentiment-Analysis
pip install -r requirements.txt
python main.py
```

---

## 📄 Licence
Ce projet est sous licence **MIT** – vous êtes libre de le réutiliser, le modifier ou le partager avec attribution.

---

## 🙌 Auteur
Réalisé par **EL ASRI MOUAD**  
🔗 [Mon LinkedIn](https://www.linkedin.com/in/mouad-el-asri-24332b32b)  
📂 [Voir le projet sur GitHub](https://github.com/Mouad-cmd-cpu/IMDB-Sentiment-Analysis)
