import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np


pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)     # 显示所有行

#对于分类变量或者离散变量，使用平均值填充可能不是一个好的选择，通常会选择使用众数（出现次数最多的值）进行填充。
def data_processing_nomal(path):
    df = pd.read_csv(path)
    scaler = StandardScaler()#对数据进行标准化处理，使得处理后的数据符合标准正态分布，即均值为 0，标准差为 1。
                             #这样处理后，数据的中心点会在原点，且数据的分布在各个方向上的范围相同。
    features = df.iloc[:, 1:-1]
    #print(type(features))  <class 'pandas.core.frame.DataFrame'>
    scaler.fit(features)#fit方法用于计算特征的均值和标准差，这些值将用于后续的标准化（缩放）过程。
    features = scaler.transform(features)#transform方法可以正确地对数据进行标准化。
    #print(type(features))   <class 'pandas.core.frame.DataFrame'>
    features = pd.DataFrame(features)
    numeric_features = features.dtypes[features.dtypes != 'object'].index  # 获得数值型列索引
    features[numeric_features] = features[numeric_features].fillna(0)  # 用0填充数值型列的空值
    features_labels = pd.concat([features, df[['label']]], axis=1)
    df = pd.concat([df[['sample_id']], features_labels], axis=1)
    return df

def data_processing_nomal_new(df):
    scaler = StandardScaler()#对数据进行标准化处理，使得处理后的数据符合标准正态分布，即均值为 0，标准差为 1。
                             #这样处理后，数据的中心点会在原点，且数据的分布在各个方向上的范围相同。
    features = df.iloc[:, 1:-1]
    #print(type(features))  <class 'pandas.core.frame.DataFrame'>
    scaler.fit(features)#fit方法用于计算特征的均值和标准差，这些值将用于后续的标准化（缩放）过程。
    features = scaler.transform(features)#transform方法可以正确地对数据进行标准化。
    #print(type(features))   <class 'pandas.core.frame.DataFrame'>
    features = pd.DataFrame(features)
    numeric_features = features.dtypes[features.dtypes != 'object'].index  # 获得数值型列索引
    features[numeric_features] = features[numeric_features].fillna(0)  # 用0填充数值型列的空值
    #features_labels = pd.concat([features, df[['label']]], axis=1)
    df = pd.concat([df[['sample_id']], features], axis=1)
    return df



#print(data_processing_nomal_new())


def data_processing_nomal_test(path):
    df = pd.read_csv(path)
    scaler = StandardScaler()  # 对数据进行标准化处理，使得处理后的数据符合标准正态分布，即均值为 0，标准差为 1。
    # 这样处理后，数据的中心点会在原点，且数据的分布在各个方向上的范围相同。
    features = df.iloc[:, 1:]
    # print(type(features))  <class 'pandas.core.frame.DataFrame'>
    scaler.fit(features)  # fit方法用于计算特征的均值和标准差，这些值将用于后续的标准化（缩放）过程。
    features = scaler.transform(features)  # transform方法可以正确地对数据进行标准化。
    # print(type(features))   <class 'pandas.core.frame.DataFrame'>
    features = pd.DataFrame(features)
    numeric_features = features.dtypes[features.dtypes != 'object'].index  # 获得数值型列索引
    features[numeric_features] = features[numeric_features].fillna(0)  # 用0填充数值型列的空值
    #features_labels = pd.concat([features, df[['label']]], axis=1)
    df = pd.concat([df[['sample_id']], features], axis=1)
    return df


def data_processing_min_max(path):
    df = pd.read_csv(path)
    min_max = MinMaxScaler()
    features = df.iloc[:,1:-1]
    min_max.fit(features)
    features = min_max.transform(features)
    features = pd.DataFrame(features)
    numeric_features = features.dtypes[features.dtypes != 'object'].index  # 获得数值型列索引

    #features[numeric_features] = features[numeric_features].fillna()  # 用均值填充数值型列的空值,可能有点问题
    for column in numeric_features:
        features[column] = features[column].fillna(features[column].mode().iloc[0],inplace=True)  # 用众数填充数值型列的空值
        #注意mode().iloc[0]是因为mode()可能会返回多个众数，我们通常使用第一个众数进行替换。
    features_labels = pd.concat([features, df[['label']]], axis=1)
    df = pd.concat([df[['sample_id']], features_labels], axis=1)
    return df

def corr():
    data = pd.read_csv('训练数据集/preprocess_train.csv')
    corr_col = data.corr()['label'].abs() > 0.01  # 处理相关值大于0.01的列
    corr_col = corr_col.values.reshape(1, -1)
    corr_col = corr_col.reshape(-1)
    corr_col = corr_col[1:]
    # 进行降维操作，并且将sample_id删掉
    return corr_col

def data_processing_corr(path):

    df = data_processing_nomal(path)

    corr_col = corr()
    Sample_id = df.iloc[:, 0]
    df = df.drop('sample_id', axis=1)
    df = df.iloc[:, corr_col]
    df = pd.concat([Sample_id, df], axis=1)
    return df

def data_solve_train_valid(path):
    train = data_processing_nomal('E:\杂项\软件杯大赛\训练数据集\preprocess_train.csv')
    valid = data_processing_nomal('E:\\杂项\\软件杯大赛\\验证集\\validate_1000.csv')
    features1 = train.iloc[:, 1:]
    features2 = valid.iloc[:, 1:]
    df = pd.concat([features1, features2], axis=0)
    df = df.reset_index(drop=True)#drop=True避免将索引列产生新的一列。
    df = df.reset_index()
    df.rename(columns={'index': 'sample_id'}, inplace=True)
    scaler = StandardScaler()  # 对数据进行标准化处理，使得处理后的数据符合标准正态分布，即均值为 0，标准差为 1。
    # 这样处理后，数据的中心点会在原点，且数据的分布在各个方向上的范围相同。
    features = df.iloc[:, 1:-1]

    scaler.fit(features)  # fit方法用于计算特征的均值和标准差，这些值将用于后续的标准化（缩放）过程。
    features = scaler.transform(features)  # transform方法可以正确地对数据进行标准化。

    features = pd.DataFrame(features)
    features_labels = pd.concat([features, df[['label']]], axis=1)
    df = pd.concat([df[['sample_id']], features_labels], axis=1)
    return df

def data_solve_RFECV(data_path,k):
    df = data_processing_nomal(data_path)
    feature_importances = np.load(r'E:\杂项\软件杯大赛\Catboost\feature_importances.npy',allow_pickle=True)
    top_k_idx = np.argsort(feature_importances)[-k:]
    X = df.drop(['sample_id', 'label'], axis=1)
    #y = df[['label']]
    X = X.iloc[:, top_k_idx]
    X = pd.concat([X,df[['label']]],axis=1)
    df = pd.concat([df[['sample_id']],X],axis = 1)
    return df

def data_solve_RFECV_2(data_path,k):
    df = data_processing_nomal(data_path)
    feature_importances = np.loadtxt(r'D:\软件杯大赛\Catboost\Features_importance.csv')
    top_k_idx = np.argsort(feature_importances)[-k:]
    X = df.drop(['sample_id', 'label'], axis=1)
    #y = df[['label']]
    X = X.iloc[:, top_k_idx]
    X = pd.concat([X,df[['label']]],axis=1)
    df = pd.concat([df[['sample_id']],X],axis = 1)
    return df



def data_solve_RFECV_test(data_path,k):
    df = data_processing_nomal_test(data_path)
    feature_importances = np.loadtxt(r'D:\软件杯大赛\Catboost\Features_importance.csv')
    top_k_idx = np.argsort(feature_importances)[-k:]
    X = df.drop(['sample_id'], axis=1)
    X = X.iloc[:, top_k_idx]
    df = pd.concat([df[['sample_id']], X], axis=1)
    return df


# def data_processing_nomal(path):
#     df = pd.read_csv(path)
#     features = df.iloc[:,1:-1]
#     numeric_features = features.dtypes[features.dtypes != 'object'].index  # 获得数值型列索引
#     features[numeric_features] = features[numeric_features].apply(
#         lambda x: (x - x.mean()) / (x.std())
#     )
#     features[numeric_features] = features[numeric_features].fillna(0)  # 是一个函数，它会查找所选列中的所有空值（NaN），并用0来填充这些空值。
#     features_labels = pd.concat([features, df[['label']]], axis=1)
#     df = pd.concat([df[['sample_id']], features_labels], axis=1)
#     return df


if __name__ == "__main__":# 只有在脚本直接执行时才会执行的代码块
    # df = data_solve_RFECV_2("训练数据集/preprocess_train.csv",21)
    # # print(df.shape)
    # # print(df.describe())
    # print(df.head())
    #
    # df = data_solve_RFECV("训练数据集/preprocess_train.csv", 21)
    # # print(df.shape)
    # # print(df.describe())
    # print(df.head())

    df = data_solve_RFECV_test(r'D:\软件杯大赛\测试集\test_2000_x.csv',21)
    print(df.head())