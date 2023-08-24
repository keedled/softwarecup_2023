import catboost
from catboost import CatBoostClassifier,Pool
from sklearn.model_selection import train_test_split
import numpy as np
from data_solve import data_processing_nomal,data_solve_train_valid,data_solve_RFECV,data_solve_RFECV_test
#from Predict import MacroF1
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, precision_score, recall_score

def MacroF1(report):
    prec_avg = report['macro avg']['precision']
    reca_avg = report['macro avg']['recall']
    macro_F1 = (2 * prec_avg * reca_avg) / (prec_avg + reca_avg)
    return macro_F1


#25 : 86.47%
#26 : 86.46%
#27 : 86.44%
#28 : 86.54%
#29 ：86.51%
#30 : 86.34%
# X = data.drop(['sample_id', 'label'], axis=1)
# y = data[['label']]
# x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.225, random_state=52,shuffle=True,stratify=y)
#
#class_weights = {0:1,  1:4.908, 2:3.19, 3:5.86, 4:9.12, 5:6.90}#设置类的权重，一般来说类别较少的类应该有更高的权重。
# 创建和训练模型
model = CatBoostClassifier(
        random_strength = 8,
        n_estimators = 787,
        colsample_bylevel = 0.7134197145457797,
        depth =  5,
        learning_rate = 0.2767789755159874,
        l2_leaf_reg = 3.0217858996548883,
        random_seed = 65,
        boosting_type = 'Ordered',
        bootstrap_type = 'Bernoulli',
        subsample = 0.7793679490463288,
        #class_weights = 'auto',
        # auto_class_weights='Balanced'
)
train_filename = r"E:\杂项\软件杯大赛\训练数据集\preprocess_train.csv"
train = data_solve_RFECV(train_filename,21)
x_train = train.drop(['sample_id', 'label'], axis=1)
y_train = train[['label']]

test_filename = r'D:\软件杯大赛\测试集\test_2000_x.csv'
test = data_solve_RFECV_test(test_filename,61)
x_test = test.drop(['sample_id'],axis = 1)
# x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.225, random_state=42, shuffle=True,
#                                                           stratify=y)
#x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42,shuffle=True,stratify=y)
model.fit(x_train, y_train, verbose=100)
test_predict = model.predict(x_test)
# print(type(test_predict))
print(test_predict)
#test_pred_proba = model.predict_proba(x_valid)
count = []
for i in range(6):
    count.append(np.count_nonzero(test_predict == i))
    print("类别{}的数量：{}".format(i,count[i]))
print("数据总数：{}".format(np.size(test_predict)))
#print(test_pred_proba)
#report_dict = classification_report(y_valid, test_predict, output_dict=True)


# max_score = 0.8
# for i in range(10,100):
#     data = data_solve_RFECV(data_filename,i)
#     X = data.drop(['sample_id', 'label'], axis=1)
#     y = data[['label']]
#     x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.225, random_state=42, shuffle=True,
#                                                           stratify=y)
# #x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42,shuffle=True,stratify=y)
# model.fit(x_train, y_train, verbose=100)
# test_predict = model.predict(x_valid)
# test_pred_proba = model.predict_proba(x_valid)
# report_dict = classification_report(y_valid, test_predict, output_dict=True)
#     if MacroF1(report_dict) > max_score:
#         print("i = {}".format(i))
#         max_score = MacroF1(report_dict)

# print("MacroF1:{}".format(MacroF1(report_dict)))
# print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_valid,test_predict))
# report = classification_report(y_valid,test_predict)
# print(report)

# i = 10
# MacroF1:0.8811655749312332

# i = 17
# MacroF1:0.883139392554645

# i = 21
# MacroF1:0.8872495705732131

# i = 22
# MacroF1:0.8875589157620463

# i = 61
# MacroF1:0.8910740803331992