import math
import pymorphy2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
with open('Congratulations.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Инициализация лемматизатора и списка стоп-слов
stop_words = set(stopwords.words('russian'))
morph = pymorphy2.MorphAnalyzer()


def preprocess_text(text):
    # Токенизация
    tokens = word_tokenize(text.lower())
    # Удаление стоп-слов и лемматизация
    tokens = [morph.parse(word)[0].normal_form for word in tokens if word.isalnum(
    ) and word not in stop_words]
    return tokens


# Пример использования
processed_tokens = preprocess_text(text)


def build_vocabulary(corpus):
    # Создание словаря уникальных слов
    vocab = set()
    for text in corpus:
        vocab.update(text)
    return list(vocab)


def bow_vectorize(text, vocab):
    # Создание вектора BoW
    vector = [0] * len(vocab)
    word_counts = defaultdict(int)
    for word in text:
        word_counts[word] += 1
    for i, word in enumerate(vocab):
        vector[i] = word_counts.get(word, 0)
    return vector


corpus = processed_tokens.copy()

vocab = build_vocabulary(corpus)
print("Vocabulary:", vocab)

with open('bag_of_word.txt', 'w', encoding='utf-8') as file:
    for text in corpus:
        vector = bow_vectorize(text, vocab)
        file.write(f"Text: {text} -> BoW Vector: {vector} \n")


def compute_tf(text, vocab):
    # Вычисление Term Frequency (TF)
    tf = [0] * len(vocab)
    word_counts = defaultdict(int)
    for word in text:
        word_counts[word] += 1
    for i, word in enumerate(vocab):
        tf[i] = word_counts.get(word, 0) / len(text)
    return tf


def compute_idf(corpus, vocab):
    # Вычисление Inverse Document Frequency (IDF)
    idf = [0] * len(vocab)
    total_docs = len(corpus)
    for i, word in enumerate(vocab):
        doc_count = sum(1 for text in corpus if word in text)
        idf[i] = math.log((total_docs + 1) / (doc_count + 1)) + 1
    return idf


def compute_tfidf(corpus, vocab):
    # Вычисление TF-IDF
    tfidf_vectors = []
    idf = compute_idf(corpus, vocab)
    for text in corpus:
        tf = compute_tf(text, vocab)
        tfidf = [tf[i] * idf[i] for i in range(len(vocab))]
        tfidf_vectors.append(tfidf)
    return tfidf_vectors


with open('tf_idf.txt', 'w', encoding='utf-8') as file:
    tfidf_vectors = compute_tfidf(corpus, vocab)
    for i, vector in enumerate(tfidf_vectors):
        file.write(f"Text {i+1} -> TF-IDF Vector: {vector} \n")

with open('лемстопток.txt', 'w', encoding='utf-8') as file:
    file.write(' '.join(map(str, processed_tokens)))
