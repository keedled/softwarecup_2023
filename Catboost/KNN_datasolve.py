import pandas as pd
from sklearn.impute import KNNImputer as KNN
import numpy as np
from sklearn.preprocessing import StandardScaler


def data_processing_KNN(path,k):
    df = pd.read_csv(path)
    features = df.iloc[:, 1:-1]
    knnImpute = KNN(n_neighbors=k)
    features = pd.DataFrame(knnImpute.fit_transform(features))
    print(features.head())
    scaler = StandardScaler()
    scaler.fit(features)#fit方法用于计算特征的均值和标准差，这些值将用于后续的标准化（缩放）过程。
    features = scaler.transform(features)#transform方法可以正确地对数据进行标准化。
    features = pd.DataFrame(features)
    features_labels = pd.concat([features, df[['label']]], axis=1)
    df = pd.concat([df[['sample_id']], features_labels], axis=1)
    return df


#df = data_processing_KNN('E:\杂项\软件杯大赛\训练数据集\preprocess_train.csv')
#print(df.describe())