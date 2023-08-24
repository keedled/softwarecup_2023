import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np


path = r'D:\软件杯大赛\验证集\validate_1000.csv'
df = pd.read_csv(path)
print(type(df))
scaler = StandardScaler()  # 对数据进行标准化处理，使得处理后的数据符合标准正态分布，即均值为 0，标准差为 1。
# 这样处理后，数据的中心点会在原点，且数据的分布在各个方向上的范围相同。
features = df.iloc[:, 1:-1]
# print(type(features))  <class 'pandas.core.frame.DataFrame'>
scaler.fit(features)  # fit方法用于计算特征的均值和标准差，这些值将用于后续的标准化（缩放）过程。
features = scaler.transform(features)  # transform方法可以正确地对数据进行标准化。
# print(type(features))   <class 'pandas.core.frame.DataFrame'>
features = pd.DataFrame(features)
numeric_features = features.dtypes[features.dtypes != 'object'].index  # 获得数值型列索引
features[numeric_features] = features[numeric_features].fillna(0)  # 用0填充数值型列的空值
features_labels = pd.concat([features, df[['label']]], axis=1)
df = pd.concat([df[['sample_id']], features_labels], axis=1)