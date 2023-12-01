import pandas as pd
import io


# 使用 pandas 读取数据
df = pd.read_csv('/Users/chenqiaoling/Desktop/blog/codes/LinS/all2all_64.csv', delim_whitespace=True)

print(df.head())  # 展示前几行数据以检查结果是否正确

