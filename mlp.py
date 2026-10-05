# Multi-Layer Perceptron
# 依照片還原。原程式約第 1–4 行不在畫面中，第 5 行僅見一個未成對的 """，此處略去。
import numpy as np
import math
import random
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "PMingLiU"]
plt.rcParams["axes.unicode_minus"] = False

# *****建立資料, Initialization
xs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
for i in range(4):  # 列印輸入矩陣
    for j in range(2):
        print(float(xs[i][j]), end=" ")
    print()
yd = [0.0, 0.0, 0.0, 0.0]  # 目標矩陣
w = [0.5, 0.4, 0.9, 1.0, -1.2, 1.1]  # w13, w23, w14, w24, w35, w45
ya = [0.0, 0.0, 0.0]  # Y3, Y4, Y5 三個神經元的輸出
theta = [0.8, -0.1, 0.3]  # T3, T4, T5 三個神經元的門檻值
alpha = 0.2  # Learning rate，執行時可改為 0.1 到 0.9
x = np.arange(-1, 3, 0.01)  # 繪圖範圍
epouch = 0  # 循環次數
training_times = 0  # 訓練次數
MSElist = []


def line1(A):  # *****自行設計直線函數, Y3
    return (-A * w[0] + theta[0]) / w[1]


def line2(A):  # *****自行設計直線函數, Y4
    return (-A * w[2] + theta[1]) / w[3]


def axis(A):  # ******設計軸線函數
    return 0 * A


# ********* Weight 取亂數值
for i in range(6):
    w[i] = random.random()
    print("w%d=%f" % (i, w[i]))
# *** 請使用者輸入資料
print("*****This is a Multi-Layer Perceptron learning program****")
select = eval(input("請輸入: 1<AND> or 2<OR> or 3<XOR>: "))
fig, (line_ax, mse_ax) = plt.subplots(1, 2, figsize=(12, 5))
if select == 1:
    yd = [0.0, 0.0, 0.0, 1.0]
    line_ax.plot([0], [0], "rx")  # 繪製四點用紅色區分
    line_ax.plot([0], [1], "rx")  # 繪製四點用紅色區分
    line_ax.plot([1], [0], "rx")  # 繪製四點用紅色區分
    line_ax.plot([1], [1], "bo")  # 繪製四點用藍色區分
if select == 2:
    yd = [0.0, 1.0, 1.0, 1.0]
    line_ax.plot([0], [0], "rx")  # 繪製四點用紅色區分
    line_ax.plot([0], [1], "bo")  # 繪製四點用藍色區分
    line_ax.plot([1], [0], "bo")  # 繪製四點用藍色區分
    line_ax.plot([1], [1], "bo")  # 繪製四點用藍色區分
if select == 3:
    yd = [0.0, 1.0, 1.0, 0.0]
    line_ax.plot([0], [0], "rx")  # 繪製四點用紅色區分
    line_ax.plot([0], [1], "bo")  # 繪製四點用藍色區分
    line_ax.plot([1], [0], "bo")  # 繪製四點用藍色區分
    line_ax.plot([1], [1], "rx")  # 繪製四點用紅色區分
print("** Your learning target is\n", yd)
line_ax.plot(x, line1(x), "c--")  # 畫初始設定直線, Y3
line_ax.plot(x, line2(x), "c--")  # 畫初始設定直線, Y4
line_ax.plot(x, axis(x), "k-")  # draw X軸
line_ax.plot(axis(x), x, "k-")  # draw Y軸
# ******* training loop
SE = 0.0  # square error
Accuracy = 0.0  # Mean Square error
MSE = 1.0
while True:
    try:
        alpha = float(input("Learning rate? Ex:0.2 (0.1-0.9): "))
    except ValueError:
        print("請輸入數字。")
        continue
    if 0.1 <= alpha <= 0.9:
        break
    print("請輸入 0.1 到 0.9 的學習率。")
print("Learning rate =", alpha)

Accuracy = eval(input("What is the target MSE? (例如 0.01、0.001、0.0001): "))
print(f"目標 MSE: {Accuracy}")

while MSE > Accuracy and epouch < 100000:  # step 4, Iteration
    epouch += 1
    print("epouch=", epouch)
    SE = 0.0
    for i in range(4):  # **讀四筆資料
        # step 2a
        # *****feed forward          # calculate Y3 and Y4
        ya[0] = 1.0 / (1.0 + math.exp(-(xs[i][0] * w[0] + xs[i][1] * w[1] - theta[0])))  # Y3
        ya[1] = 1.0 / (1.0 + math.exp(-(xs[i][0] * w[2] + xs[i][1] * w[3] - theta[1])))  # Y4
        # step 2b
        # calculate Y5
        ya[2] = 1.0 / (1.0 + math.exp(-(ya[0] * w[4] + ya[1] * w[5] - theta[2])))  # Y5
        print("Y3=%f,Y4=%f,Y5=%f" % (ya[0], ya[1], ya[2]))
        # step 3a
        e = yd[i] - ya[2]  # calculate real error
        ge5 = ya[2] * (1.0 - ya[2]) * e  # calculate gradient error of Y5
        # **** change weights (learning)
        # *****backward propagation
        w[4] = w[4] + alpha * ya[0] * ge5  # change w35
        w[5] = w[5] + alpha * ya[1] * ge5  # change w45
        theta[2] = theta[2] + (-alpha) * ge5  # change theta3
        # step 3b
        # ***** error backward calculation ** this is most important part
        ge3 = ya[0] * (1.0 - ya[0]) * ge5 * w[4]  # gradient error of Y3 with error-back
        ge4 = ya[1] * (1.0 - ya[1]) * ge5 * w[5]  # gradient error of Y4 with error-back
        # **** change weights (learning)
        w[0] = w[0] + alpha * xs[i][0] * ge3  # change w13
        w[1] = w[1] + alpha * xs[i][1] * ge3  # change w23
        w[2] = w[2] + alpha * xs[i][0] * ge4  # change w14
        w[3] = w[3] + alpha * xs[i][1] * ge4  # change w24
        theta[0] = theta[0] + (-alpha) * ge3  # change theta1
        theta[1] = theta[1] + (-alpha) * ge4  # change theta2
        SE += e * e  # accumulate square error
    MSE = SE / 4.0  # 計算每次循環之均方誤差值
    print("MSE=", MSE)
    MSElist.append(MSE)
print("******劃出調整後的紅色直線,青色為原始設定線*******")
line_ax.plot(x, line1(x), "r-")  # 畫調整後直線 Y3
line_ax.plot(x, line2(x), "r--")  # 畫調整後直線 Y4
line_ax.set_xlabel("---X1---")  # X座標軸說明
line_ax.set_ylabel("---X2---")  # Y座標軸說明
line_ax.set_title("決策線 - 初始設定(青色) vs 訓練後(紅色)")
mse_ax.plot(range(epouch), MSElist)
mse_ax.set_xlabel("---EPOUCH No.---")  # X座標軸說明
mse_ax.set_ylabel("---MSE: Mean Square Error---")  # Y座標軸說明
mse_ax.set_title("均方誤差 - 訓練過程")
fig.tight_layout()
plt.show()
# Program End
