import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# 创建数据
data = {
    "Bandwidth_MB_s": [41.746, 62.982, 65.596, 101.968, 138.671, 159.773, 177.197, 190.415, 193.555, 194.056, 194.097,
                       193.776, 193.419, 193.679, 194.425, 194.462, 36.732, 55.592, 80.364, 100.85, 116.875, 133.242,
                       160.23, 178.519, 189.055, 193.55, 193.752, 193.717, 193.417, 193.686, 194.365, 194.416, 33.096,
                       48.456, 72.221, 97.357, 113.762, 125.266, 134.315, 164.453, 178.744, 187.352, 192.915, 193.512,
                       192.669, 193.47, 194.342, 194.218],
    "Cards": [64] * 16 + [128] * 16 + [256] * 16,
    "Data_MB": [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304,
                8388608, 16777216] * 3
}

# 转换为DataFrame
df = pd.DataFrame(data)

# 可视化数据
plt.figure(figsize=(12, 6))
for card in df['Cards'].unique():
    subset = df[df['Cards'] == card]
    plt.scatter(subset['Data_MB'], subset['Bandwidth_MB_s'], label=f'{card} cards')

plt.xlabel('Data Transferred (MB)')
plt.ylabel('Effective Bandwidth (MB/s)')
plt.title('Effective Bandwidth vs Data Transferred for Different Card Numbers')
plt.legend()
plt.xscale('log')
plt.grid(True)
plt.show()

# 显示前几行数据以验证
df.head()

# 准备多项式回归模型
degree = 3  # 多项式的度数
poly_features = PolynomialFeatures(degree=degree, include_bias=False)

# 用于存储拟合结果和评分
models = {}
scores = {}

# 分别对每个卡数的数据进行拟合
for card in df['Cards'].unique():
    subset = df[df['Cards'] == card]

    # 准备数据
    X = subset['Data_MB'].values.reshape(-1, 1)
    y = subset['Bandwidth_MB_s'].values
    X_poly = poly_features.fit_transform(X)

    # 拟合模型
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)

    # 评估模型
    score = r2_score(y, y_pred)

    # 存储模型和评分
    models[card] = model
    scores[card] = score

    # 可视化拟合结果
    plt.scatter(X, y, label=f'{card} cards')
    plt.plot(X, y_pred, label=f'{card} cards Fit')

# 绘制图表
plt.xlabel('Data Transferred (MB)')
plt.ylabel('Effective Bandwidth (MB/s)')
plt.title('Polynomial Regression Fit for Different Card Numbers')
plt.xscale('log')
plt.legend()
plt.grid(True)
plt.show()

# 输出评分
scores

