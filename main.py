import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
import warnings
warnings.filterwarnings('ignore')

# Fonction pour créer un dossier s'il n'existe pas
def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Dossier créé: {directory_path}")
    return directory_path

# Création des répertoires pour sauvegarder les visualisations et résultats ML
create_directory_if_not_exists('visualisation')
create_directory_if_not_exists('visualisation/raw')
create_directory_if_not_exists('visualisation/preprocessed')
create_directory_if_not_exists('ML')
create_directory_if_not_exists('ML/ini')
create_directory_if_not_exists('ML/improved')

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)

# Load the dataset
df = pd.read_csv("IMDB Dataset.csv")
print("Aperçu des données:")
print(df.head(4))
print("\nDescription des données:")
print(df.describe())
print("\nInformations sur les données:")
print(df.info())

#####################################
# PARTIE 1: VISUALISATION DES DONNÉES BRUTES
#####################################
print("\n--- VISUALISATION DES DONNÉES BRUTES ---")

# Distribution des sentiments
plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment', data=df)
plt.title('Distribution des Sentiments')
plt.savefig('visualisation/raw/sentiment_distribution.png')
plt.close()

# Analyse de la longueur des critiques
df['review_length'] = df['review'].apply(len)
plt.figure(figsize=(12, 6))
sns.histplot(df['review_length'], bins=50, kde=True)
plt.title('Distribution des Longueurs de Critiques')
plt.xlabel('Longueur de Critique (caractères)')
plt.savefig('visualisation/raw/review_length_distribution.png')
plt.close()

# Comparaison des longueurs par sentiment
plt.figure(figsize=(12, 6))
sns.boxplot(x='sentiment', y='review_length', data=df)
plt.title('Longueur des Critiques par Sentiment')
plt.savefig('visualisation/raw/review_length_by_sentiment.png')
plt.close()

# Nuage de mots pour les critiques positives
positive_reviews = ' '.join(df[df['sentiment'] == 'positive']['review'].sample(1000, random_state=42))
wordcloud_positive = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(positive_reviews)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title('Nuage de Mots pour les Critiques Positives')
plt.savefig('visualisation/raw/positive_wordcloud.png')
plt.close()

# Nuage de mots pour les critiques négatives
negative_reviews = ' '.join(df[df['sentiment'] == 'negative']['review'].sample(1000, random_state=42))
wordcloud_negative = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(negative_reviews)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title('Nuage de Mots pour les Critiques Négatives')
plt.savefig('visualisation/raw/negative_wordcloud.png')
plt.close()

print("Visualisations des données brutes enregistrées dans 'visualisation/raw/'")

#####################################
# PARTIE 2: PRÉTRAITEMENT DES DONNÉES
#####################################
print("\n--- PRÉTRAITEMENT DES DONNÉES ---")

# Fonction de prétraitement
def preprocess_text(text):
    # Suppression des balises HTML
    text = re.sub('<.*?>', ' ', text)
    
    # Suppression des caractères non alphabétiques
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    # Conversion en minuscules
    text = text.lower()
    
    # Tokenisation
    words = text.split()
    
    # Suppression des stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    # Rejoindre les mots en texte
    return ' '.join(words)

# Application du prétraitement aux critiques
print("Prétraitement des critiques en cours...")
df['processed_review'] = df['review'].apply(preprocess_text)
print("Prétraitement terminé!")

# Affichage d'un échantillon de critiques prétraitées
print("\nÉchantillon de critiques prétraitées:")
print(df[['review', 'processed_review']].head(3))

# Sauvegarde des données prétraitées
df.to_csv('preprocessed_imdb.csv', index=False)
print("Données prétraitées sauvegardées dans 'preprocessed_imdb.csv'")

#####################################
# PARTIE 3: VISUALISATION DES DONNÉES PRÉTRAITÉES
#####################################
print("\n--- VISUALISATION DES DONNÉES PRÉTRAITÉES ---")

# Longueur des critiques prétraitées
df['processed_length'] = df['processed_review'].apply(len)
plt.figure(figsize=(12, 6))
sns.histplot(df['processed_length'], bins=50, kde=True)
plt.title('Distribution des Longueurs de Critiques Prétraitées')
plt.xlabel('Longueur (caractères)')
plt.savefig('visualisation/preprocessed/processed_length_distribution.png')
plt.close()

# Comparaison des longueurs avant/après prétraitement
plt.figure(figsize=(12, 6))
sns.boxplot(data=pd.DataFrame({
    'Original': df['review_length'],
    'Prétraité': df['processed_length']
}))
plt.title('Comparaison des Longueurs Avant/Après Prétraitement')
plt.savefig('visualisation/preprocessed/length_comparison.png')
plt.close()

# Nuage de mots pour les critiques positives prétraitées
positive_processed = ' '.join(df[df['sentiment'] == 'positive']['processed_review'].sample(1000, random_state=42))
wordcloud_positive_processed = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(positive_processed)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_positive_processed, interpolation='bilinear')
plt.axis('off')
plt.title('Nuage de Mots pour les Critiques Positives Prétraitées')
plt.savefig('visualisation/preprocessed/positive_processed_wordcloud.png')
plt.close()

# Nuage de mots pour les critiques négatives prétraitées
negative_processed = ' '.join(df[df['sentiment'] == 'negative']['processed_review'].sample(1000, random_state=42))
wordcloud_negative_processed = WordCloud(width=800, height=400, background_color='white', max_words=200).generate(negative_processed)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_negative_processed, interpolation='bilinear')
plt.axis('off')
plt.title('Nuage de Mots pour les Critiques Négatives Prétraitées')
plt.savefig('visualisation/preprocessed/negative_processed_wordcloud.png')
plt.close()

# Mots les plus fréquents dans les critiques positives et négatives
def get_top_words(sentiment_type):
    text = ' '.join(df[df['sentiment'] == sentiment_type]['processed_review'])
    words = text.split()
    word_counts = pd.Series(words).value_counts().head(20)
    return word_counts

plt.figure(figsize=(12, 8))
get_top_words('positive').plot(kind='bar')
plt.title('Top 20 des Mots dans les Critiques Positives')
plt.savefig('visualisation/preprocessed/top_positive_words.png')
plt.close()

plt.figure(figsize=(12, 8))
get_top_words('negative').plot(kind='bar')
plt.title('Top 20 des Mots dans les Critiques Négatives')
plt.savefig('visualisation/preprocessed/top_negative_words.png')
plt.close()

print("Visualisations des données prétraitées enregistrées dans 'visualisation/preprocessed/'")

#####################################
# PARTIE 4: ENTRAÎNEMENT INITIAL DES MODÈLES
#####################################
print("\n--- ENTRAÎNEMENT INITIAL DES MODÈLES ---")

# Préparation des données pour l'entraînement
# Conversion des étiquettes en valeurs numériques
df['sentiment_numeric'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Extraction des caractéristiques avec Bag of Words
print("Création des caractéristiques Bag of Words...")
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['processed_review'])
y = df['sentiment_numeric']

# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Dimensions des données d'entraînement: {X_train.shape}")
print(f"Dimensions des données de test: {X_test.shape}")

# Fonction pour évaluer et visualiser les performances d'un modèle
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, save_dir):
    # S'assurer que le répertoire de sauvegarde existe
    create_directory_if_not_exists(save_dir)
    
    # Entraînement du modèle
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Évaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Affichage des résultats
    print(f"\nRésultats pour {model_name}:")
    print(f"Précision: {accuracy:.4f}")
    print(report)
    
    # Visualisation de la matrice de confusion
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Négatif', 'Positif'], 
                yticklabels=['Négatif', 'Positif'])
    plt.xlabel('Prédiction')
    plt.ylabel('Réalité')
    plt.title(f'Matrice de Confusion - {model_name}')
    plt.savefig(f'{save_dir}/{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()
    
    return accuracy, model

# Entraînement de différents modèles
models = {
    'Naive Bayes': MultinomialNB(),
    'Régression Logistique': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM Linéaire': LinearSVC(random_state=42)
}

results = {}
for name, model in models.items():
    accuracy, trained_model = evaluate_model(model, X_train, X_test, y_train, y_test, name, 'ML/ini')
    results[name] = accuracy

# Comparaison des performances des modèles
plt.figure(figsize=(12, 6))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title('Comparaison des Performances des Modèles')
plt.ylabel('Précision')
plt.ylim(0.7, 1.0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('ML/ini/model_comparison.png')
plt.close()

print("Résultats de l'entraînement initial enregistrés dans 'ML/ini/'")

#####################################
# PARTIE 5: AMÉLIORATION DES MODÈLES
#####################################
print("\n--- AMÉLIORATION DES MODÈLES ---")

# Utilisation de TF-IDF au lieu de Bag of Words
print("Création des caractéristiques TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(max_features=10000, min_df=5, max_df=0.8)
X_tfidf = tfidf_vectorizer.fit_transform(df['processed_review'])

# Division en ensembles d'entraînement et de test
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Modèles améliorés avec hyperparamètres optimisés
improved_models = {
    'Naive Bayes (TF-IDF)': MultinomialNB(alpha=0.1),
    'Régression Logistique (TF-IDF)': LogisticRegression(C=10, max_iter=1000, random_state=42),
    'Random Forest (TF-IDF)': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
    'SVM Linéaire (TF-IDF)': LinearSVC(C=1.0, random_state=42)
}

improved_results = {}
for name, model in improved_models.items():
    accuracy, trained_model = evaluate_model(model, X_train_tfidf, X_test_tfidf, y_train, y_test, name, 'ML/improved')
    improved_results[name] = accuracy

# Comparaison des performances des modèles améliorés
plt.figure(figsize=(14, 6))
sns.barplot(x=list(improved_results.keys()), y=list(improved_results.values()))
plt.title('Comparaison des Performances des Modèles Améliorés')
plt.ylabel('Précision')
plt.ylim(0.7, 1.0)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('ML/improved/improved_model_comparison.png')
plt.close()

# Comparaison des modèles initiaux vs améliorés
best_initial = max(results.items(), key=lambda x: x[1])
best_improved = max(improved_results.items(), key=lambda x: x[1])

print(f"\nMeilleur modèle initial: {best_initial[0]} avec une précision de {best_initial[1]:.4f}")
print(f"Meilleur modèle amélioré: {best_improved[0]} avec une précision de {best_improved[1]:.4f}")

# Visualisation de la comparaison
comparison_data = pd.DataFrame({
    'Modèle': ['Meilleur Initial', 'Meilleur Amélioré'],
    'Précision': [best_initial[1], best_improved[1]],
    'Nom': [best_initial[0], best_improved[0]]
})

plt.figure(figsize=(10, 6))
bars = sns.barplot(x='Modèle', y='Précision', data=comparison_data)
plt.title('Comparaison du Meilleur Modèle Initial vs Amélioré')
plt.ylim(0.8, 1.0)

# Ajouter les noms des modèles sur les barres
for i, bar in enumerate(bars.patches):
    bars.text(bar.get_x() + bar.get_width()/2., 
              bar.get_height() + 0.005, 
              comparison_data['Nom'].iloc[i], 
              ha='center', va='bottom', rotation=0, size=10)

plt.tight_layout()
plt.savefig('ML/improved/best_model_comparison.png')
plt.close()

print("Résultats de l'entraînement amélioré enregistrés dans 'ML/improved/'")
print("\nTraitement terminé avec succès!")