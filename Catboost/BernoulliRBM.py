from sklearn.neural_network import BernoulliRBM
import catboost
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_solve import data_processing_nomal
from sklearn.metrics import classification_report


train_filename = r"E:\杂项\软件杯大赛\训练数据集\preprocess_train.csv"
train = data_processing_nomal(train_filename)
test_filename = 'E:\\杂项\\软件杯大赛\\验证集\\validate_1000.csv'
test = data_processing_nomal(test_filename)

x_train = train.drop(['sample_id', 'label'], axis=1)
y_train = train[['label']]
x_valid = test.drop(['sample_id','label'],axis = 1)
y_valid = test[['label']]


# 使用RBM进行特征提取
rbm = BernoulliRBM(n_components=300)
x_train_transformed = rbm.fit_transform(x_train)
x_valid_transformed = rbm.transform(x_valid)

# 使用CatBoost进行分类
model = CatBoostClassifier(loss_function='MultiClass',n_estimators=1000)
train_pool = catboost.Pool(x_train_transformed,y_train)
model.fit(train_pool, verbose=10)


def MacroF1(report):
    prec_avg = report['macro avg']['precision']
    reca_avg = report['macro avg']['recall']
    macro_F1 = (2 * prec_avg * reca_avg) / (prec_avg + reca_avg)
    return macro_F1

# 预测测试集
test_predict = model.predict(x_valid_transformed)

#test_pred_proba = model.predict_proba(X_test)
#print(y_pred_proba)

print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_valid,test_predict))
report = classification_report(y_valid,test_predict)
report_dict = classification_report(y_valid, test_predict,output_dict=True)

print(report)
print("MacroF1:{:.2f}%".format(MacroF1(report_dict) * 100))

