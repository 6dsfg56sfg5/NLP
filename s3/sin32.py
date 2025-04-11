import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class NeuralNetwork:
    def __init__(self, input_size: int = 1, hidden_neurons: int = 4):
        self.input_size = input_size
        self.hidden_neurons = hidden_neurons
        # Инициализация Xavier/Glorot
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_neurons) * np.sqrt(
            2. / (self.input_size + self.hidden_neurons))
        self.weights_hidden_output = np.random.randn(
            self.hidden_neurons, 1) * np.sqrt(2. / (self.hidden_neurons + 1))
        self.bias_hidden = np.zeros((1, self.hidden_neurons))
        self.bias_output = np.zeros((1, 1))

    def activation_func(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x))

    def forward(self, X: np.ndarray) -> np.ndarray:
        hidden_layer = self.activation_func(
            np.dot(X, self.weights_input_hidden) + self.bias_hidden)
        output_layer = np.dot(
            hidden_layer,
            self.weights_hidden_output) + self.bias_output
        return output_layer

    def train(
            self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 1000,
            lr: float = 0.001,
            batch_size: int = 64):
        # Стандартизация данных
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42)

        losses = []
        for epoch in range(epochs):
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Прямое распространение
                hidden_layer = self.activation_func(
                    np.dot(X_batch, self.weights_input_hidden) + self.bias_hidden)
                output_layer = np.dot(
                    hidden_layer, self.weights_hidden_output) + self.bias_output

                # Обратное распространение
                output_error = y_batch - output_layer
                d_output = output_error  # Для линейного выхода производная = 1

                hidden_error = np.dot(d_output, self.weights_hidden_output.T)
                d_hidden = hidden_error * (hidden_layer * (1 - hidden_layer))

                # Обновление параметров
                self.weights_hidden_output += lr * \
                    np.dot(hidden_layer.T, d_output) / batch_size
                self.bias_output += lr * \
                    np.sum(d_output, axis=0, keepdims=True) / batch_size
                self.weights_input_hidden += lr * \
                    np.dot(X_batch.T, d_hidden) / batch_size
                self.bias_hidden += lr * \
                    np.sum(d_hidden, axis=0, keepdims=True) / batch_size

            # Проверка ошибки
            test_output = self.forward(X_test)
            test_loss = np.mean(np.square(y_test - test_output))
            losses.append(test_loss)

            if epoch % 100 == 0:
                log_message = f"Epoch {epoch}, Test Loss: {test_loss:.6f}\n"
                with open('output2.txt', 'a') as f:
                    f.write(log_message)

        return losses


if __name__ == "__main__":
    df = pd.read_csv('diamond.csv')
    df = df.rename(columns={'Carat Weight': 'Carat'})
    df = df.dropna()

    X = df['Price'].values.reshape(-1, 1)
    y = df['Carat'].values.reshape(-1, 1)

    nn = NeuralNetwork(input_size=1, hidden_neurons=8)
    losses = nn.train(X, y, epochs=2000, lr=0.001)
