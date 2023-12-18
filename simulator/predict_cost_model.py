import functools
import os
import pickle
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

import profiler.benchmark
from profiler.benchmark.multi_head_attn import UnitMultiHeadAttn
from profiler.profiler import run_profile
from utils.common import MB, CostType
from utils.config import Config


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
            segments = {"all": (0, float("inf"))}
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

    def predict(self, world_size, complexity):
        try:
            model = self.model_fit[self.return_segments(complexity)][world_size]
            X_pred = self.poly_features.fit_transform([[complexity]])
            Y_pred = model.predict(X_pred)[0]
            return Y_pred
        except Exception as e:
            print(f"e: {e}", flush=True)
            import pdb

            pdb.set_trace()


class SplineModel:
    def __init__(self):
        self._data_prefix = "data"
        self.spline_model_list = {}
        self.data = {}
        self.load_data()
        self.build_model()

    def load_data(self):
        for cost_data_file in os.listdir(self._data_prefix):
            name, suffix = cost_data_file.split(".")
            if suffix == "pickle":
                with open(f"{self._data_prefix}/{cost_data_file}", "rb") as f:
                    self.data[name] = pickle.load(f)

    def build_model(self):
        for cost_type in self.data.keys():
            if cost_type != CostType.FLASH_ATTN:  # fa我们直接查表，不预测
                for world_size in self.data[cost_type].keys():
                    data = self.data[cost_type][world_size]
                    x = data["Data_B"]
                    y = data["Latency_s"]
                    self.spline_model_list[cost_type] = {}
                    self.spline_model_list[cost_type][world_size] = interp1d(x, y, kind="slinear")
            else:
                self.spline_model_list[cost_type] = {}
                self.spline_model_list[cost_type][1] = self.data[cost_type][1]

    def predict(self, cost_type, world_size, complexity):
        return self.spline_model_list[cost_type][world_size](complexity)

    def predict_cost(self, cost_type: CostType, complexity=0, world_size=1, **kwargs):
        """predict computation cost
        The cost of attention will use KV mapping, and the cost of linear will
        use PolynomialModel.

        Args:
            cost_type (CostType): _description_
            complexity (int, optional): _description_. Defaults to 0.

        Returns:
            float: op latency.
        """
        if cost_type == CostType.FLASH_ATTN:
            try:
                return self.spline_model_list[cost_type][1][UnitMultiHeadAttn.gen_store_key(**kwargs)][0]["lat"]
            except KeyError as e:
                import pdb

                pdb.set_trace()
        else:
            spline_model = self.spline_model_list[cost_type][world_size]
            try:
                predict = spline_model(complexity)
            except ValueError as e:
                below_bounds, above_bounds = spline_model.x[0], spline_model.x[-1]
                if complexity < below_bounds:
                    return spline_model(below_bounds)  # 如果超过下界就返回下界
                if complexity > above_bounds:
                    lat = spline_model(above_bounds)
                    return lat * complexity / above_bounds  # 如果超过上界就线性扩展
                raise ValueError(f"value error for cost_type:{cost_type}, complexity:{complexity}")
            else:
                return predict


def my_compare(a, b):
    world_size_a, complexity_a = a[0], a[2]
    world_size_b, complexity_b = b[0], b[2]
    # print(world_size_a, world_size_b, complexity_a, complexity_b)

    if world_size_a > world_size_b:
        return True
    elif world_size_a < world_size_b:
        return False
    else:
        if complexity_a > complexity_b:
            return True
        elif complexity_a < complexity_b:
            return False
        else:
            assert ValueError, f"a:{a}, b:{b}"


class GenCostModel:
    def __init__(self, is_master=True, re_build_cost_data=False, build_type_list=None) -> None:
        self._master = is_master
        self._profile_args = Config(
            {
                "trials": 10,
                "warmups": 1,
            }
        )
        self.cost_data = None
        self._data_prefix = "./data"
        self.cost_kv_data = {}
        self.build_cost_model_by_key_value(build_type_list)

    def _log(self, msg: str):
        if self._master:
            print(msg, flush=True)

    def build_cost_model_by_key_value(self, build_type_list):
        if self.cost_data is None:
            self.cost_data = OrderedDict()
            for bench_type in build_type_list:
                self._log(f"now test {bench_type}")
                re_results = run_profile(self._profile_args, bench_type)
                with open(f"{self._data_prefix}/{bench_type}.pickle", "wb+") as f:
                    pickle.dump(re_results, f)

    @staticmethod
    def reformat_data_to_cost_model(total_results):
        list_data = []

        for world_size in total_results.keys():
            for complexity in total_results[world_size].keys():
                for value in total_results[world_size][complexity]:
                    print(value)
                    list_data.append([world_size, value["lat"], complexity])

        list_data.sort(key=functools.cmp_to_key(my_compare))
        data_list = list(map(list, zip(*list_data)))
        data = {"World_Size": data_list[0], "Latency_ms": data_list[1], "Data_MB": data_list[2]}

        return data


cost_model = None


def get_predict_or_kv_cost(cost_type: CostType, complexity=0, **kwargs):
    global cost_model
    if cost_model is None:
        cost_model = SplineModel()

    return cost_model.predict_cost(cost_type, complexity=complexity, **kwargs)
