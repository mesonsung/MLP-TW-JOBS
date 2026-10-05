"""多層感知器：隱藏層可設 1 到 3 層，用反向傳播學習 AND、OR 或 XOR。"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np

# 預設 DejaVu Sans 沒有中文，標題會缺字並跳出 Glyph warning。
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "PMingLiU"]
plt.rcParams["axes.unicode_minus"] = False

LEARNING_RATE = 0.2
LEARNING_RATE_RANGE = (0.1, 0.9)
MAX_EPOCHS = 100_000
HIDDEN_LAYER_CHOICES = (1, 2, 3)
NEURON_COUNT_RANGE = (1, 20)
INPUT_SIZE = 2
PLOT_X = np.arange(-1, 3, 0.01)

# 四筆二元輸入：(x1, x2)
SAMPLES = ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0))
GATES = {
    1: ("AND", (0.0, 0.0, 0.0, 1.0)),
    2: ("OR", (0.0, 1.0, 1.0, 1.0)),
    3: ("XOR", (0.0, 1.0, 1.0, 0.0)),
}

# 每一層兩個隱藏神經元的門檻，沿用原程式
HIDDEN_THRESHOLDS = (0.8, -0.1)
OUTPUT_THRESHOLD = 0.3


def random_weights(count):
    """正負都有的初始權重，決策線才分得開，多層時也比較不容易卡住。"""
    return [random.uniform(-1.0, 1.0) for _ in range(count)]


def sigmoid(net):
    """sigmoid(net) = 1 / (1 + e^(-net))。正負分開計算，避免 exp 溢位。"""
    if net >= 0:
        z = math.exp(-net)
        return 1.0 / (1.0 + z)
    z = math.exp(net)
    return z / (1.0 + z)


def sigmoid_gradient(output):
    """sigmoid 輸出對淨輸入的導數：y * (1 - y)。"""
    return output * (1.0 - output)


class Neuron:
    """單一 sigmoid 神經元。淨輸入 = 權重 · 輸入 - 門檻。"""

    def __init__(self, weights, threshold):
        self.weights = list(weights)
        self.threshold = threshold

    def forward(self, inputs):
        net = sum(weight * value for weight, value in zip(self.weights, inputs))
        return sigmoid(net - self.threshold)

    def apply_delta(self, inputs, delta, rate):
        for index, value in enumerate(inputs):
            self.weights[index] += rate * value * delta
        self.threshold += -rate * delta

    def decision_line(self, x1):
        """第一隱藏層（兩個原始輸入）的決策線：w0*x1 + w1*x2 - threshold = 0。"""
        weight_x1, weight_x2 = self.weights
        return (-weight_x1 * x1 + self.threshold) / weight_x2


def hidden_threshold(index):
    """前兩個神經元沿用原程式的 0.8、-0.1，其餘從 0 開始。"""
    if index < len(HIDDEN_THRESHOLDS):
        return HIDDEN_THRESHOLDS[index]
    return 0.0


class MultiLayerPerceptron:
    """輸入 2 維，輸出 1 個。隱藏層 1 到 3 層，每層神經元數量可設定。"""

    def __init__(self, hidden_layer_count, neurons_per_layer, learning_rate=LEARNING_RATE):
        if hidden_layer_count not in HIDDEN_LAYER_CHOICES:
            raise ValueError("隱藏層只能是 1、2 或 3 層")
        low, high = NEURON_COUNT_RANGE
        if neurons_per_layer < low or neurons_per_layer > high:
            raise ValueError(f"每層神經元數量要在 {low} 到 {high} 之間")
        rate_low, rate_high = LEARNING_RATE_RANGE
        if learning_rate < rate_low or learning_rate > rate_high:
            raise ValueError(f"學習率要在 {rate_low} 到 {rate_high} 之間")
        self.learning_rate = learning_rate
        self.neurons_per_layer = neurons_per_layer
        self.hidden_layers = []
        fan_in = INPUT_SIZE
        for _ in range(hidden_layer_count):
            layer = [
                Neuron(random_weights(fan_in), hidden_threshold(index))
                for index in range(neurons_per_layer)
            ]
            self.hidden_layers.append(layer)
            fan_in = len(layer)
        self.output = Neuron(random_weights(fan_in), OUTPUT_THRESHOLD)
        self.layers = [*self.hidden_layers, [self.output]]

    def describe_weights(self):
        rows = []
        sources = [f"x{index}" for index in range(1, INPUT_SIZE + 1)]
        for layer_index, layer in enumerate(self.hidden_layers, start=1):
            next_sources = []
            for neuron_index, neuron in enumerate(layer, start=1):
                name = f"H{layer_index}.{neuron_index}"
                next_sources.append(name)
                for source, weight in zip(sources, neuron.weights):
                    rows.append((f"{source} -> {name}", weight))
                rows.append((f"threshold {name}", neuron.threshold))
            sources = next_sources
        for source, weight in zip(sources, self.output.weights):
            rows.append((f"{source} -> Y", weight))
        rows.append(("threshold Y", self.output.threshold))
        return rows

    def forward_output(self, sample):
        activations = self._activations(sample)
        return activations[-1][0]

    def _activations(self, sample):
        """activations[i] 是第 i 層的輸入，最後一項是輸出層的輸出。"""
        activations = [list(sample)]
        for layer in self.layers:
            activations.append([neuron.forward(activations[-1]) for neuron in layer])
        return activations

    def learn(self, sample, target):
        """對一筆資料做前向計算與反向傳播，回傳輸出誤差。"""
        activations = self._activations(sample)
        prediction = activations[-1][0]
        error = target - prediction
        deltas = [sigmoid_gradient(prediction) * error]
        rate = self.learning_rate

        for layer_index in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[layer_index]
            incoming = activations[layer_index]
            # 先用更新前的權重把 delta 傳回上一層，再更新這一層。
            if layer_index > 0:
                previous_deltas = propagate_deltas(layer, incoming, deltas)
            else:
                previous_deltas = None
            for neuron, delta in zip(layer, deltas):
                neuron.apply_delta(incoming, delta, rate)
            deltas = previous_deltas
        return error

    def train(self, samples, targets, mse_target, max_epochs=MAX_EPOCHS):
        history = []
        mse = math.inf
        epoch = 0
        while mse > mse_target and epoch < max_epochs:
            epoch += 1
            square_error = 0.0
            for sample, target in zip(samples, targets):
                error = self.learn(sample, target)
                square_error += error * error
            mse = square_error / len(samples)
            history.append(mse)
            print(f"epoch={epoch}  MSE={mse}")
        return history

    def first_layer_lines(self):
        return [neuron.decision_line(PLOT_X) for neuron in self.hidden_layers[0]]


def propagate_deltas(next_layer, outputs, next_deltas):
    """用下一層目前的權重，計算這一層每個神經元的 delta。"""
    deltas = []
    for index, activation in enumerate(outputs):
        combined = sum(
            neuron.weights[index] * delta
            for neuron, delta in zip(next_layer, next_deltas)
        )
        deltas.append(sigmoid_gradient(activation) * combined)
    return deltas


def ask_choice(prompt, options):
    while True:
        raw = input(prompt).strip()
        if raw in options:
            return raw
        print(f"請輸入 {'、'.join(options)}。")


def ask_int(prompt, low, high):
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("請輸入整數。")
            continue
        if value < low or value > high:
            print(f"請輸入 {low} 到 {high} 的整數。")
            continue
        return value


def ask_float(prompt, low, high):
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print("請輸入數字。")
            continue
        if value < low or value > high:
            print(f"請輸入 {low} 到 {high} 的數值。")
            continue
        return value


def ask_positive_float(prompt):
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print("請輸入數字。")
            continue
        if value <= 0:
            print("請輸入大於 0 的數值。")
            continue
        return value


def print_samples(samples):
    print("輸入資料:")
    for sample in samples:
        print(" ".join(f"{value:.1f}" for value in sample))


def print_network(model):
    hidden = " -> ".join(
        f"H{index}({len(layer)})"
        for index, layer in enumerate(model.hidden_layers, start=1)
    )
    print(f"網路結構: 輸入({INPUT_SIZE}) -> {hidden} -> 輸出(1)")
    print(f"學習率: {model.learning_rate}")
    print("初始權重:")
    for name, value in model.describe_weights():
        print(f"  {name} = {value:.6f}")


def output_grid(model):
    axis = np.linspace(-1, 3, 120)
    xx, yy = np.meshgrid(axis, axis)
    zz = np.empty(xx.shape)
    for row in range(xx.shape[0]):
        for col in range(xx.shape[1]):
            zz[row, col] = model.forward_output((xx[row, col], yy[row, col]))
    return xx, yy, zz


def plot_result(model, samples, targets, initial_lines, trained_lines, mse_history):
    xx, yy, zz = output_grid(model)
    plt.contourf(xx, yy, zz, levels=20, cmap="coolwarm", alpha=0.45)
    plt.colorbar(label="網路輸出")

    for (x1, x2), target in zip(samples, targets):
        marker = "ro" if target == 1.0 else "bo"
        plt.plot([x1], [x2], marker)

    for line in initial_lines:
        plt.plot(PLOT_X, line, "c--")
    trained_styles = ("-", "--", "-.", ":")
    trained_colors = ("r", "m", "g", "C1", "C4", "C5", "C6", "C8", "C9", "brown")
    for index, line in enumerate(trained_lines):
        plt.plot(
            PLOT_X,
            line,
            color=trained_colors[index % len(trained_colors)],
            linestyle=trained_styles[index % len(trained_styles)],
        )
    plt.plot(PLOT_X, np.zeros_like(PLOT_X), "k-")
    plt.plot(np.zeros_like(PLOT_X), PLOT_X, "k-")
    plt.xlim(-1, 3)
    plt.ylim(-1, 3)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("第一隱藏層決策線　青：訓練前　紅：訓練後　背景：網路輸出")
    plt.show()

    plt.plot(range(1, len(mse_history) + 1), mse_history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.show()


def print_predictions(model, samples, targets):
    print("訓練後輸出:")
    for sample, target in zip(samples, targets):
        prediction = model.forward_output(sample)
        print(f"  {sample} -> {prediction:.4f}  (目標 {target:.0f})")


def main():
    print("This is a Multi-Layer Perceptron learning program")
    print_samples(SAMPLES)

    depth_text = ask_choice("隱藏層數 1、2 或 3: ", {"1", "2", "3"})
    low, high = NEURON_COUNT_RANGE
    neurons = ask_int(f"每層隱藏神經元數量（{low} 到 {high}）: ", low, high)
    rate_low, rate_high = LEARNING_RATE_RANGE
    learning_rate = ask_float(
        f"學習率（{rate_low} 到 {rate_high}，例如 {LEARNING_RATE}）: ",
        rate_low,
        rate_high,
    )
    model = MultiLayerPerceptron(int(depth_text), neurons, learning_rate)
    print_network(model)

    choice = ask_choice("請輸入 1<AND>、2<OR> 或 3<XOR>: ", {"1", "2", "3"})
    gate_name, targets = GATES[int(choice)]
    print(f"學習目標 {gate_name}: {list(targets)}")

    initial_lines = model.first_layer_lines()
    mse_target = ask_positive_float("目標 MSE（例如 0.01、0.001、0.0001）: ")
    mse_history = model.train(SAMPLES, targets, mse_target)
    print_predictions(model, SAMPLES, targets)

    print("劃出第一隱藏層決策線，青色虛線為訓練前，紅色為訓練後")
    plot_result(model, SAMPLES, targets, initial_lines, model.first_layer_lines(), mse_history)


if __name__ == "__main__":
    main()
