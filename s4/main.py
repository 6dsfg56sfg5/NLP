from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.decomposition import PCA


# Загрузка датасета
newsgroups = fetch_20newsgroups(
    subset='all', remove=(
        'headers', 'footers', 'quotes'))
texts = newsgroups.data
labels = newsgroups.target
target_names = newsgroups.target_names

print(f"Количество текстов: {len(texts)}")
print(f"Количество категорий: {len(target_names)}")


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def preprocess_text(text):
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление спецсимволов и чисел
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Удаление стоп-слов
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Лемматизация
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


# Применяем предобработку ко всем текстам
processed_texts = [preprocess_text(text) for text in texts]


# Анализ распределения классов
plt.figure(figsize=(10, 6))
plt.hist(labels, bins=20, rwidth=0.8)
plt.xticks(range(20), target_names, rotation=90)
plt.title('Распределение текстов по категориям')
plt.show()

# Анализ длины текстов
text_lengths = [len(text.split()) for text in processed_texts]
plt.figure(figsize=(10, 6))
plt.hist(text_lengths, bins=50)
plt.title('Распределение длины текстов')
plt.xlabel('Количество слов')
plt.ylabel('Частота')
plt.show()


# Создание TF-IDF векторов
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(processed_texts)


# Кластеризация K-means
kmeans = KMeans(n_clusters=20, random_state=42)
clusters = kmeans.fit_predict(X)

# Сравнение с реальными метками
ari = adjusted_rand_score(labels, clusters)
silhouette = silhouette_score(X, clusters)

print(f"Adjusted Rand Index: {ari:.3f}")
print(f"Silhouette Score: {silhouette:.3f}")

# Визуализация с помощью PCA


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X.toarray())

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                      c=clusters, cmap='viridis', alpha=0.5)
plt.title('Визуализация кластеров (PCA)')
plt.colorbar(scatter)
plt.show()


# Сначала разделим на train+val и test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, labels, test_size=0.15, random_state=42, stratify=labels)

# Затем разделим train+val на train и val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15 / 0.85, random_state=42, stratify=y_temp)

print(f"Train size: {X_train.shape[0]}")
print(f"Val size: {X_val.shape[0]}")
print(f"Test size: {X_test.shape[0]}")
