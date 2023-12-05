from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

from utils.common import *


class PolynomialModel:
    def __init__(self, degree, data, name="unkonw", segments=None) -> None:
        """_summary_

        Args:
            degree (int): _description_
            data (dict): _description_
            segments (dict): _description_
        """
        self.name = name
        self.degree = 3  # 多项式的度数
        self.poly_features = PolynomialFeatures(degree=degree, include_bias=False)  # 准备多项式回归模型
        self.data = pd.DataFrame(data)  # 转换为DataFrame
        if segments is None:
            segments = {"all": (min(data["Data_MB"]), max(data["Data_MB"]))}
        print(segments, flush=True)
        self.segments = OrderedDict(segments)
        self.segment_scores = {seg: {} for seg in self.segments}  # 用于存储拟合结果和评分
        self.model_fit = {
            seg: {card: None for card in self.data["World_Size"].unique()} for seg in self.segments
        }  # 存储模型
        self.see_base_value()
        self.build_model()

    def see_base_value(self):
        # 可视化数据
        plt.figure(figsize=(12, 6))
        for card in self.data["World_Size"].unique():
            subset = self.data[self.data["World_Size"] == card]
            plt.scatter(subset["Data_MB"], subset["Latency_ms"], label=f"{card} cards")

        plt.xlabel("Data Transferred (MB)")
        plt.ylabel("Latency (ms)")
        plt.title("Transferred Latency vs Data Transferred for Different Card Numbers")
        plt.legend()
        plt.xscale("log")
        plt.grid(True)
        plt.savefig(f"{self.name}.jpg")
        plt.show()
        print(self.data.head())

    def build_model(self):
        # 对每个分段和卡数的数据进行拟合
        plt.figure(figsize=(12, 6))
        for seg, (low, high) in self.segments.items():
            for card in self.data["World_Size"].unique():
                subset = self.data[
                    (self.data["World_Size"] == card) & (self.data["Data_MB"] >= low) & (self.data["Data_MB"] < high)
                ]

                # 如果该段中没有足够的数据点，则跳过
                if len(subset) < 2:
                    continue

                # 准备数据
                X = subset["Data_MB"].values.reshape(-1, 1)
                y = subset["Latency_ms"].values
                X_poly = self.poly_features.fit_transform(X)

                # 拟合模型
                model = LinearRegression()
                model.fit(X_poly, y)
                y_pred = model.predict(X_poly)
                self.model_fit[seg][card] = model

                # 评估模型
                score = r2_score(y, y_pred)
                self.segment_scores[seg][card] = score

                # 可视化拟合结果
                plt.scatter(X / MB, y, label=f"{card} cards")
                plt.plot(X / MB, y_pred, label=f"{card} cards Fit")

        # 绘制图表
        plt.xlabel("Data Transferred (MB)")
        plt.ylabel("Latency (ms)")
        plt.title("Segmented Polynomial Regression Fit for Different Card Numbers")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.name}_fit.jpg")
        plt.show()

    def return_segments(self, x):
        for key, value in self.segments.items():
            low, hight = value[0], value[1]
            if x >= low and x < hight:
                return key
        assert ValueError, f"predict value:{x} out of range"

    def predict(self, x):
        predictions = {}
        for card, model in self.model_fit[self.return_segments(x)].items():
            X_pred = self.poly_features.fit_transform([[x]])
            predictions[card] = model.predict(X_pred)[0]
            print(f"X: {pretty_print_size(x)} cards: {card}, latency: {predictions[card]:.3f} ms")
        return predictions
