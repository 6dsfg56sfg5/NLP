import pandas as pd
import numpy as np


class NeuralNetwork:
    def __init__(self, hidden_neurons: int):
        self.hidden_neurons = hidden_neurons
        self.neurons = []  # Список для хранения нейронов (их коэффициентов)

    def _linear_regression(self, X: np.ndarray, y: np.ndarray) -> list[float]:
        coefficients = np.polyfit(X, y, 1)
        return coefficients

    def _activation_func(
            self,
            x: list[float],
            coefficients: list[float]) -> list[float]:
        w, b = coefficients
        return [w * xi + b for xi in x]

    def train(self, X: pd.Series, y: pd.Series):
        self.neurons = []
        for _ in range(self.hidden_neurons):
            # Для каждого нейрона вычисляем свои коэффициенты
            coefficients = self._linear_regression(X.values, y.values)
            self.neurons.append(coefficients)

    def predict(self, X: pd.Series) -> list[float]:
        if not self.neurons:
            raise ValueError("Еррор! Модель не обучена!!!")

        # Усредняем предсказания всех нейронов
        predictions = np.zeros(len(X))
        for neuron_coeffs in self.neurons:
            neuron_pred = self._activation_func(X.values, neuron_coeffs)
            predictions += np.array(neuron_pred)

        return (predictions / len(self.neurons)).tolist()


if __name__ == "__main__":
    # Загрузка данных
    df = pd.read_csv('diamond.csv')
    df = df.dropna()

    nn = NeuralNetwork(hidden_neurons=228)

    # Обучение модели
    nn.train(df['Price'], df['Carat Weight'])

    # Предсказание
    predictions = nn.predict(df['Price'])

    # Сохранение нормализованных результатов
    with open('output2.txt', 'w', encoding='utf-8') as file:
        file.write(str(predictions))
