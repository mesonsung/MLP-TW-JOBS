"""2-2-1 多層感知器：輸入層、隱藏層、輸出層分開，學習 AND、OR 或 XOR。"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "PMingLiU"]
plt.rcParams["axes.unicode_minus"] = False

LEARNING_RATE = 0.2
MAX_EPOCHS = 100_000
PLOT_X = np.arange(-1, 3, 0.01)

# 輸入層的四筆資料：(x1, x2)
SAMPLES = ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0))
GATES = {
    "1": ("AND", (0.0, 0.0, 0.0, 1.0), ("bo", "bo", "bo", "ro")),
    "2": ("OR", (0.0, 1.0, 1.0, 1.0), ("ro", "bo", "bo", "bo")),
    "3": ("XOR", (0.0, 1.0, 1.0, 0.0), ("ro", "bo", "bo", "ro")),
}


def sigmoid(net):
    return 1.0 / (1.0 + math.exp(-net))


def sigmoid_gradient(output):
    return output * (1.0 - output)


class Neuron:
    """一個 sigmoid 神經元。淨輸入 = 權重 · 輸入 - 門檻。"""

    def __init__(self, fan_in, threshold):
        self.weights = [random.random() for _ in range(fan_in)]
        self.threshold = threshold
        self.output = 0.0

    def forward(self, inputs):
        net = sum(weight * value for weight, value in zip(self.weights, inputs))
        self.output = sigmoid(net - self.threshold)
        return self.output

    def apply(self, inputs, delta, rate):
        for index, value in enumerate(inputs):
            self.weights[index] += rate * value * delta
        self.threshold += -rate * delta


class InputLayer:
    """沒有權重，只提供 (x1, x2)。"""

    def __init__(self, samples):
        self.samples = samples


class HiddenLayer:
    """兩個神經元 Y3、Y4，接收輸入層。門檻沿用原程式 0.8、-0.1。"""

    def __init__(self):
        self.neurons = (Neuron(2, 0.8), Neuron(2, -0.1))

    def forward(self, sample):
        return [neuron.forward(sample) for neuron in self.neurons]

    def backward(self, sample, outputs, output_delta, output_weights, rate):
        # output_weights 是輸出層剛剛更新後的權重，與原程式相同。
        deltas = [
            sigmoid_gradient(output) * output_delta * weight
            for output, weight in zip(outputs, output_weights)
        ]
        for neuron, delta in zip(self.neurons, deltas):
            neuron.apply(sample, delta, rate)

    def decision_lines(self, x1):
        lines = []
        for neuron in self.neurons:
            weight_x1, weight_x2 = neuron.weights
            lines.append((-weight_x1 * x1 + neuron.threshold) / weight_x2)
        return lines


class OutputLayer:
    """一個神經元 Y5，接收隱藏層。門檻沿用原程式 0.3。"""

    def __init__(self):
        self.neuron = Neuron(2, 0.3)

    def forward(self, hidden_outputs):
        return self.neuron.forward(hidden_outputs)

    def backward(self, hidden_outputs, target, rate):
        error = target - self.neuron.output
        delta = sigmoid_gradient(self.neuron.output) * error
        self.neuron.apply(hidden_outputs, delta, rate)
        return error, delta


class MultiLayerPerceptron:
    def __init__(self, learning_rate=LEARNING_RATE):
        self.learning_rate = learning_rate
        self.input_layer = InputLayer(SAMPLES)
        self.hidden_layer = HiddenLayer()
        self.output_layer = OutputLayer()

    def learn(self, sample, target):
        hidden = self.hidden_layer.forward(sample)
        self.output_layer.forward(hidden)
        error, output_delta = self.output_layer.backward(
            hidden, target, self.learning_rate
        )
        self.hidden_layer.backward(
            sample,
            hidden,
            output_delta,
            self.output_layer.neuron.weights,
            self.learning_rate,
        )
        return error

    def train(self, targets, mse_target, max_epochs=MAX_EPOCHS):
        history = []
        mse = math.inf
        epoch = 0
        while mse > mse_target and epoch < max_epochs:
            epoch += 1
            square_error = 0.0
            for sample, target in zip(self.input_layer.samples, targets):
                error = self.learn(sample, target)
                square_error += error * error
            mse = square_error / len(self.input_layer.samples)
            history.append(mse)
            print(f"epoch={epoch}  MSE={mse}")
        return history


def ask_choice(prompt, options):
    while True:
        raw = input(prompt).strip()
        if raw in options:
            return raw
        print(f"請輸入 {'、'.join(options)}。")


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


def print_network(model):
    print("輸入資料:")
    for sample in model.input_layer.samples:
        print(" ".join(f"{value:.1f}" for value in sample))

    print("初始權重:")
    y3, y4 = model.hidden_layer.neurons
    links = (
        ("w13", "x1 -> Y3", y3.weights[0]),
        ("w23", "x2 -> Y3", y3.weights[1]),
        ("w14", "x1 -> Y4", y4.weights[0]),
        ("w24", "x2 -> Y4", y4.weights[1]),
        ("w35", "Y3 -> Y5", model.output_layer.neuron.weights[0]),
        ("w45", "Y4 -> Y5", model.output_layer.neuron.weights[1]),
    )
    for name, meaning, value in links:
        print(f"  {name} ({meaning}) = {value:.6f}")


def plot_result(samples, targets, colors, initial_lines, trained_lines, mse_history):
    _, (lines_ax, mse_ax) = plt.subplots(1, 2, figsize=(12, 5))

    for (x1, x2), color in zip(samples, colors):
        lines_ax.plot([x1], [x2], color)
    for line in initial_lines:
        lines_ax.plot(PLOT_X, line, "c--")
    lines_ax.plot(PLOT_X, trained_lines[0], "r-")
    lines_ax.plot(PLOT_X, trained_lines[1], "r--")
    lines_ax.plot(PLOT_X, np.zeros_like(PLOT_X), "k-")
    lines_ax.plot(np.zeros_like(PLOT_X), PLOT_X, "k-")
    lines_ax.set_xlim(-1, 3)
    lines_ax.set_ylim(-1, 3)
    lines_ax.set_xlabel("X1")
    lines_ax.set_ylabel("X2")
    lines_ax.set_title("青色虛線：訓練前    紅色：訓練後")

    mse_ax.plot(range(1, len(mse_history) + 1), mse_history)
    mse_ax.set_xlabel("Epoch")
    mse_ax.set_ylabel("MSE")
    mse_ax.set_title("Mean Square Error")

    plt.tight_layout()
    plt.show()


def main():
    print("This is a Multi-Layer Perceptron learning program")
    model = MultiLayerPerceptron()
    print_network(model)

    choice = ask_choice("請輸入 1<AND>、2<OR> 或 3<XOR>: ", set(GATES))
    gate_name, targets, colors = GATES[choice]
    print(f"學習目標 {gate_name}: {list(targets)}")

    initial_lines = model.hidden_layer.decision_lines(PLOT_X)
    mse_target = ask_positive_float("目標 MSE（例如 0.01、0.001、0.0001）: ")
    mse_history = model.train(targets, mse_target)

    print("劃出調整後的紅色直線，青色虛線為訓練前的決策線")
    trained_lines = model.hidden_layer.decision_lines(PLOT_X)
    plot_result(SAMPLES, targets, colors, initial_lines, trained_lines, mse_history)


if __name__ == "__main__":
    main()
