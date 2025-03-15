import json
import chardet
import nltk
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from nltk.stem import SnowballStemmer


# Инициализация лемматизатора и стеммера
morph = MorphAnalyzer()
stemmer = SnowballStemmer("russian")

with open('Congratulations.txt', 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']

# print(f"Определена кодировка: {encoding}")


# Чтение текста из файла
with open('Congratulations.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Токенизация текста (разделение на буквы)

text = nltk.word_tokenize(text)
# находим русские стопслова
stop_words = set(stopwords.words('russian'))

# Лемматизация
lemmatized_words = [morph.parse(
    word)[0].normal_form for word in text if word.lower() not in stop_words]

# Стемминг
stemmed_words = [stemmer.stem(word)
                 for word in text if word.lower() not in stop_words]

# Сохранение результатов лемматизации в файл
with open('lemmatized_output.txt', 'w', encoding='utf-8') as file:
    file.write(' '.join(lemmatized_words))

# Сохранение результатов стемминга в файл
with open('stemmed_output.txt', 'w', encoding='utf-8') as file:
    file.write(' '.join(stemmed_words))

print("Лемматизация и стемминг завершены. Результаты сохранены в файлы.")

# буду использовать ord(char) который  возвращает ASCII-код символа

result = ''.join(text)


def tokenezation(text):
    tok = []
    for i in text:
        tok.append(i)
    return tok


def vectorization(text: str) -> dict[int]:
    return {i: word for i, word in enumerate(text)}


lemmatized_words = ''.join(lemmatized_words)
tokenization_text = tokenezation(lemmatized_words)

vectorization_text = vectorization(tokenization_text)

vec = vectorization(result)
# Сохранение результатов 2 functions in the files so good and testy
with open('tokenization.txt', 'w', encoding='utf-8') as file:
    file.write(' '.join(map(str, tokenization_text)))

# Сохранение результатов стемминга в файл

with open('vectorization.json', 'w', encoding='utf-8') as file:
    json.dump(vec, file)
