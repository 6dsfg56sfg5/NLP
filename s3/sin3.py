import pandas as pd
import numpy as np


# Линейная регрессия
def linear_regression(df: pd.DataFrame) -> list[float]:
    X = df['Price'].values
    y = df['Carat Weight'].values
    # Вычисление коэффициентов линейной регрессии (y = wx + b)
    coefficients = np.polyfit(X, y, 1)
    return coefficients

# Функция активации (линейная)


def activation_func(x: list[float], coefficients: list[float]) -> list[float]:
    w, b = coefficients
    return [w * xi + b for xi in x]

# Нейрон


def neuron(df: pd.DataFrame) -> list[float]:
    coefficients = linear_regression(df)
    result = activation_func(df['Price'].values, coefficients)
    return result


df = pd.read_csv('\\NLP\\s3\\diamond.csv')
# print(df.info())
df_drop = df.dropna()
# Проверка на пропущенные значения
# print(df.isnull().sum()) # пропусков нет

Y = df['Carat Weight']  # выбираем целевую переменную (категориальную)
X = df['Price']  # переменные для проверки влияния

# В моем случае я дропаю базовую переменную, а не только. Y

# Пример использования

if __name__ == "__main__":
    # Получаем предсказания от нейрона

    with open('output.txt', 'w', encoding='utf-8') as file:
        predictions = neuron(df)
        file.write(str(predictions))

