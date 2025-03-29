import numpy as np
import pandas as pd
from typing import Tuple


class NeuralNetwork:
    def __init__(self, input_size: int, hidden_neurons: int):
        self.input_size = input_size
        self.hidden_neurons = hidden_neurons
        self.weights_input_hidden = np.random.randn(
            input_size, hidden_neurons) * 0.1
        self.weights_hidden_output = np.random.randn(hidden_neurons, 1) * 0.1

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        return x * (1 - x)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.hidden_layer = self.sigmoid(np.dot(X, self.weights_input_hidden))
        self.output_layer = self.sigmoid(
            np.dot(self.hidden_layer, self.weights_hidden_output))
        return self.hidden_layer, self.output_layer

    def train(
            self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 1000,
            lr: float = 0.01,
            log_interval: int = 100):
        y = y.reshape(-1, 1)

        # Открываем файл для записи логов
        with open('training_log.txt', 'w') as log_file:
            for epoch in range(epochs):
                hidden_layer, output_layer = self.forward(X)

                output_error = y - output_layer
                output_delta = output_error * \
                    self.sigmoid_derivative(output_layer)

                hidden_error = np.dot(
                    output_delta, self.weights_hidden_output.T)
                hidden_delta = hidden_error * \
                    self.sigmoid_derivative(hidden_layer)

                self.weights_hidden_output += lr * \
                    np.dot(hidden_layer.T, output_delta)
                self.weights_input_hidden += lr * np.dot(X.T, hidden_delta)

                if epoch % log_interval == 0:
                    loss = np.mean(np.square(output_error))
                    # Записываем в файл вместо вывода в консоль
                    log_file.write(f"Epoch {epoch}, Loss: {loss:.6f}\n")


if __name__ == "__main__":
    df = pd.read_csv('diamond.csv')
    df = df.dropna()

    X = df['Price'].values.reshape(-1, 1)
    y = df['Carat Weight'].values

    X = (X - X.min()) / (X.max() - X.min())
    y = (y - y.min()) / (y.max() - y.min())

    X = np.hstack([X, np.ones((X.shape[0], 1))])

    nn = NeuralNetwork(input_size=2, hidden_neurons=3)
    nn.train(X, y, epochs=1000, lr=0.1)

    _, predictions = nn.forward(X)

    results = pd.DataFrame({
        'Actual': df['Carat Weight'],
        'Predicted': predictions.flatten()
    })
    results.to_csv('predictions.csv', index=False)

    np.savetxt(
        'weights_input_hidden.csv',
        nn.weights_input_hidden,
        delimiter=',')
    np.savetxt(
        'weights_hidden_output.csv',
        nn.weights_hidden_output,
        delimiter=',')

    print("Результаты обучения сохранены в training_log.txt")
    print("Предсказания сохранены в predictions.csv")
    print("Веса сети сохранены в weights_input_hidden.csv и weights_hidden_output.csv")
