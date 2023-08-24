from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from data_solve import data_processing_nomal
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
train_filename = r"E:\杂项\软件杯大赛\训练数据集\preprocess_train.csv"
train = data_processing_nomal(train_filename)
test_filename = 'E:\\杂项\\软件杯大赛\\验证集\\validate_1000.csv'
test = data_processing_nomal(test_filename)

x_train = train.drop(['sample_id', 'label'], axis=1)
y_train = train[['label']]
x_valid = test.drop(['sample_id','label'],axis = 1)
y_valid = test[['label']]

# 划分数据集为训练集和测试集

# 使用LDA进行特征提取
lda = LDA(n_components=5)  # n_components为要提取的特征数
X_train = lda.fit_transform(x_train, y_train)
X_test = lda.transform(x_valid)
print(X_train.shape)