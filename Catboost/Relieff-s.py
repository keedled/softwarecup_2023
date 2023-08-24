from skrebate import ReliefF
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import catboost
from catboost import CatBoostClassifier,Pool
from sklearn.model_selection import train_test_split
import numpy as np
from data_solve import data_processing_nomal
#from Predict import MacroF1
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, precision_score, recall_score

def MacroF1(report):
    prec_avg = report['macro avg']['precision']
    reca_avg = report['macro avg']['recall']
    macro_F1 = (2 * prec_avg * reca_avg) / (prec_avg + reca_avg)
    return macro_F1

# 假设 X 是特征矩阵，y 是目标向量
train_filename = r"E:\杂项\软件杯大赛\训练数据集\preprocess_train.csv"
train = data_processing_nomal(train_filename)
test_filename = 'E:\\杂项\\软件杯大赛\\验证集\\validate_1000.csv'
test = data_processing_nomal(test_filename)
# x_train = train.drop(['sample_id', 'label'], axis=1)
# y_train = train[['label']]
# x_test = test.drop(['sample_id','label'],axis = 1)
# y_test = test[['label']]
#y_train = y_train.flatten()
train = np.array(train)
x_train = train[:, 1:-1]
y_train = train[:, -1]
test = np.array(test)
x_test = test[:,1:-1]
y_test = test[:,-1]

# print(y_test.value_counts())
# print(type(x_train),y_train.shape)
# 特征选择
fs = ReliefF()
fs.fit(x_train, y_train)

#print(y_test.value_counts())
# 对特征按重要性进行排序，然后选择前n个最重要的特征
n = 25
importance = fs.feature_importances_
top_features = np.argsort(importance)[::-1][:n]

X_train_reduced = x_train[:, top_features]
X_test_reduced = x_test[:, top_features]

# 使用CatBoost进行训练和预测
class_weights = {0:1,  1:4.908, 2:3.19, 3:5.86, 4:9.12, 5:6.90}#设置类的权重，一般来说类别较少的类应该有更高的权重。
# 创建和训练模型
model = CatBoostClassifier(class_weights = class_weights,loss_function='MultiClass')

model.fit(X_train_reduced, y_train)

test_predict = model.predict(X_test_reduced)


print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_test,test_predict))
report = classification_report(y_test,test_predict)
report_dict = classification_report(y_test, test_predict,output_dict=True)

print(report)
print("MacroF1:{:.2f}%".format(MacroF1(report_dict) * 100))