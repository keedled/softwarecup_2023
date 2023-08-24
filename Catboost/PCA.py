# 导入所需要的库
import numpy as np
from sklearn.decomposition import PCA
from data_solve import data_processing_nomal

filename = r"E:\杂项\软件杯大赛\训练数据集\preprocess_train.csv"
data = data_processing_nomal(filename)

labels = data[['label']].values.ravel()
#print(labels.shape)
data = data.drop(['sample_id','label'],axis = 1)


# 创建PCA对象，n_components指定要保留的主成分数目
pca = PCA(n_components=2)

# 对数据进行PCA分析
data_pca = pca.fit_transform(data)

# 打印出降维后的数据
#print(data_pca)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 创建颜色列表和类别列表，为每个类别分配一个颜色
colors = ['red', 'green', 'blue', 'purple', 'yellow', 'orange']
classes = ['label 0', 'label 1', 'label 2', 'label 3', 'label 4', 'label 5']


# 在同一张图上绘制六个类别的数据
plt.figure(figsize=(10, 8))
for i, color, label in zip(range(6), colors, classes):#这是一个for循环，通过zip函数将range(6)，colors和classes组合在一起。
                                                    # 每次循环，i将从0取到5，color和label将分别取自colors和classes列表。
                                                    # 假设你有6个类别，颜色和类别名称都存储在colors和classes列表中，所以这个循环会运行6次。
    plt.scatter(data_pca[labels == i, 0], data_pca[labels == i, 1], c=color, label=label)
                                                    #这两句代码是从PCA后的数据中选择出属于第i个类别的数据。labels == i返回一个布尔数组，当标签等于i时，值为True，否则为False。
                                                    # data_pca[labels == i, 0]选择出第一主成分值属于第i类的数据，data_pca[labels == i, 1]选择出第二主成分值属于第i类的数据。
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.show()




# # 取出主成分
# x = data_pca[:, 0]  # 第一个主成分
# y = data_pca[:, 1]  # 第二个主成分
#
# # 制作散点图
# plt.scatter(x, y)
# plt.xlabel('First Principal Component')
# plt.ylabel('Second Principal Component')
# plt.show()