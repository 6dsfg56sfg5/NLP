import numpy as np
import sys


class TextGenerator:
    def __init__(self):
        self.vocab = list(
            "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ .,!?-\n")
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}

        self.hidden_size = 48
        self.vocab_size = len(self.vocab)

        self.Wxh = np.random.randn(self.hidden_size, self.vocab_size) * 0.01
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.Why = np.random.randn(self.vocab_size, self.hidden_size) * 0.01
        self.bh = np.zeros(self.hidden_size)
        self.by = np.zeros(self.vocab_size)

    def forward_pass(self, inputs, h_prev):
        """Прямой проход через сеть"""
        xs, hs, ys = {}, {}, {}
        hs[-1] = np.copy(h_prev)

        for t in range(len(inputs)):
            if inputs[t] not in self.char_to_idx:
                continue

            xs[t] = np.zeros(self.vocab_size)
            xs[t][self.char_to_idx[inputs[t]]] = 1

            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) +
                            np.dot(self.Whh, hs[t - 1]) + self.bh)
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ys[t] = np.exp(ys[t] - np.max(ys[t]))
            ys[t] /= ys[t].sum()

        return xs, hs, ys

    def backward_pass(self, inputs, targets, xs, hs, ys):
        """Обратное распространение ошибки"""
        dWxh, dWhh, dWhy = np.zeros_like(
            self.Wxh), np.zeros_like(
            self.Whh), np.zeros_like(
            self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros(self.hidden_size)

        for t in reversed(range(len(inputs))):
            if targets[t] not in self.char_to_idx:
                continue

            dy = np.copy(ys[t])
            dy[self.char_to_idx[targets[t]]] -= 1

            dWhy += np.outer(dy, hs[t])
            dby += dy

            dh = np.dot(self.Why.T, dy) + dh_next
            dh_raw = (1 - hs[t] ** 2) * dh

            dbh += dh_raw
            dWxh += np.outer(dh_raw, xs[t])
            dWhh += np.outer(dh_raw, hs[t - 1])

            dh_next = np.dot(self.Whh.T, dh_raw)

        return dWxh, dWhh, dWhy, dbh, dby

    def train_model(self, text, seq_length=25, epochs=20, lr=0.005):
        """Обучение модели"""
        original_stdout = sys.stdout
        with open('text_generation_output.txt', 'w', encoding='utf-8') as f:
            sys.stdout = f

            print("Начало обучения модели...")

            for epoch in range(epochs):
                h_prev = np.zeros(self.hidden_size)
                ptr = 0
                smooth_loss = -np.log(1.0 / self.vocab_size) * seq_length

                while ptr < len(text) - seq_length - 1:
                    inputs = text[ptr:ptr + seq_length]
                    targets = text[ptr + 1:ptr + seq_length + 1]

                    xs, hs, ys = self.forward_pass(inputs, h_prev)
                    dWxh, dWhh, dWhy, dbh, dby = self.backward_pass(
                        inputs, targets, xs, hs, ys)

                    for param, dparam in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                             [dWxh, dWhh, dWhy, dbh, dby]):
                        param -= lr * dparam

                    for t in range(len(inputs)):
                        if targets[t] in self.char_to_idx:
                            smooth_loss = smooth_loss * 0.999 + \
                                (-np.log(ys[t][self.char_to_idx[targets[t]]])) * 0.001

                    h_prev = hs[len(inputs) - 1]
                    ptr += seq_length

                if epoch % 2 == 0:
                    self.generate_text(text[:15])

            sys.stdout = original_stdout

    def generate_text(self, seed, length=150, temperature=0.7):
        """Генерация текста"""
        h = np.zeros(self.hidden_size)
        indices = [self.char_to_idx[ch]
                   for ch in seed if ch in self.char_to_idx]

        for idx in indices:
            x = np.zeros(self.vocab_size)
            x[idx] = 1
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)

        for _ in range(length):
            y = np.dot(self.Why, h) + self.by
            y = np.exp(y / temperature)
            p = y / y.sum()

            idx = np.random.choice(range(self.vocab_size), p=p)
            indices.append(idx)

            x = np.zeros(self.vocab_size)
            x[idx] = 1
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)

        result = seed + ''.join([self.idx_to_char[idx]
                                for idx in indices[len(seed):]])

        print("\n" + "=" * 60)
        print("Сгенерированный текст:")
        print(result)
        print("=" * 60)


neutral_text = """
Текст - это последовательность символов, несущая в себе информацию.
Для обработки естественного языка используются различные модели машинного обучения.
Нейронные сети могут генерировать текст, похожий на человеческий.
Чем больше данных используется для обучения, тем лучше результаты.
Генерация текста - интересная задача в области искусственного интеллекта.
""" * 30

# Запуск модели
generator = TextGenerator()
generator.train_model(neutral_text)
print("Обучение завершено! Результаты сохранены в text_generation_output.txt")
