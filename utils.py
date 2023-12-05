import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

# 可视化每个分段的拟合结果
plt.figure(figsize=(30, 20))

# Path to your Excel file
excel_path = './comm.xlsx'

# Read the Excel file
xls = pd.ExcelFile(excel_path)
ranks=[64,128,256,512]
IBs=[1,2,4]


models = {sheet_name: {rank: {ib: None for ib in IBs} for rank in ranks} for sheet_name in xls.sheet_names}

# Iterate through each sheet
for sheet_name in xls.sheet_names:
    # Read sheet into DataFrame
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    gpu64_IB1=df.iloc[0,1:]
    gpu128_IB1=df.iloc[1,1:]
    gpu256_IB1=df.iloc[2,1:]
    gpu512_IB1=df.iloc[3,1:]

    gpu64_IB2=df.iloc[6,1:]
    gpu128_IB2=df.iloc[7,1:]
    gpu256_IB2=df.iloc[8,1:]
    gpu512_IB2=df.iloc[9,1:]

    gpu64_IB4=df.iloc[12,1:]
    gpu128_IB4=df.iloc[13,1:]
    gpu256_IB4=df.iloc[14,1:]
    gpu512_IB4=df.iloc[15,1:]

    data={
        'Data_MB':[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304,
                8388608, 16777216],
        '64_IB1': gpu64_IB1,
        '128_IB1':gpu128_IB1,
        '256_IB1':gpu256_IB1,
        '512_IB1':gpu512_IB1,
        '64_IB2': gpu64_IB2,
        '128_IB2':gpu128_IB2,
        '256_IB2':gpu256_IB2,
        '512_IB2':gpu512_IB2,
        '64_IB4': gpu64_IB4,
        '128_IB4':gpu128_IB4,
        '256_IB4':gpu256_IB4,
        '512_IB4':gpu512_IB4,
    }
    for i in data:
        if i == 'Data_MB': continue
        import pdb;pdb.set_trace()
        y=np.array(data[i])
        X = np.array(data['Data_MB']).reshape(-1, 1)

        poly_features = PolynomialFeatures(degree=3, include_bias=False)
        X_poly = poly_features.fit_transform(X)

        # 从之前的模型中获取预测值
        model = LinearRegression().fit(X_poly, y)
        y_pred = model.predict(X_poly)

        models[sheet_name][i] = model

        # 绘制散点图和拟合曲线
        plt.scatter(X, y, label=f'alo: {sheet_name} & {i} ranks')
        plt.plot(X, y_pred, label=f'alo: {sheet_name} & {i} ranks - Fit')

    # Assuming the first column is the target variable and the rest are features
plt.xlabel('Data Transferred (MB)')
plt.ylabel('Latency (KB/s)')
plt.title('Segmented Polynomial Regression Fit for Different Card Numbers')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()


