import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from data_solve import data_processing_nomal, data_solve_train_valid, data_solve_train_valid, data_solve_RFECV_2, \
    data_solve_RFECV_test,data_processing_nomal_test,data_solve_RFECV_train_valid_new
#from Predict import MacroF1
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, precision_score, recall_score
from KNN_datasolve import data_processing_KNN
import json

def Create_json(test_predict,file_name):
    data_list = test_predict.tolist()
    data = {}
    for i, value in enumerate(data_list):
        key = str(i)
        value = data_list[i][0]
        data[key] = value
# 将JSON字符串写入文件
    with open("file_name", "w") as file:
        json.dump(data, file)

def MacroF1(report):
    prec_avg = report['macro avg']['precision']
    reca_avg = report['macro avg']['recall']
    macro_F1 = (2 * prec_avg * reca_avg) / (prec_avg + reca_avg)
    return macro_F1


class_weights = {0:1,  1:4.908, 2:3.19, 3:5.86, 4:9.12, 5:6.90}#设置类的权重，一般来说类别较少的类应该有更高的权重。
model = CatBoostClassifier(
                            #n_estimators = 500,
                            # learning_rate = 0.12954489804080596,
                            # random_seed = 1255,
                            #l2_leaf_reg = 0.01,
                            # # depth=5,
                            # # bagging_temperature = 2,
                            # # random_strength = 155,
                            # #per_float_feature_quantization=['10:border_count=1024','22:border_count=1024'],

                            # #per_float_feature_quantization=['10:border_count=1024'],
                            #auto_class_weights='Balanced',#MacroF1:84.63%
                            class_weights = class_weights,
                            boosting_type = 'Ordered'
                            #approx_on_full_history = True,
                            #最后两个参数可以后续测试。
)

# data_filename = r"E:\杂项\软件杯大赛\训练数据集\preprocess_train.csv"
# data = data_solve_train_valid(data_filename)

train_filename = r"E:\杂项\软件杯大赛\训练数据集\preprocess_train.csv"
#train = data_solve_RFECV_train_valid_new(100)
train = data_solve_train_valid(train_filename)
#train = data_solve_RFECV_2(train_filename,100)
valid_filename = r'D:\软件杯大赛\验证集\validate_1000.csv'
valid = data_solve_RFECV_2(valid_filename,100)

X_train = train.drop(['sample_id', 'label'], axis=1)
Y_train = train[['label']]

X_valid = valid.drop(['sample_id', 'label'], axis=1)
Y_valid = valid[['label']]

# print(X_valid[:500])
#
# print("X_train shape:", X_train.shape)
# print("Y_train shape:", X_valid.shape)
#
#
# X_train = pd.concat([X_train,X_valid[0:120]],axis=0)
# Y_train = pd.concat([Y_train,Y_valid[0:120]],axis=0)
#
# print(X_train.shape)
# print(Y_train.shape)

#X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=100)
train_pool = catboost.Pool(X_train,Y_train)
model.fit(train_pool,verbose = 100)

# train_predict = model.predict(X_test)
# print('The MacroF1 of the validate_1000 is:',metrics.accuracy_score(Y_test,train_predict))
# report = classification_report(Y_test,train_predict)
# report_dict = classification_report(Y_test,train_predict,output_dict=True)
#
# print(report)
# print("MacroF1:{:.2f}%".format(MacroF1(report_dict) * 100))


# test_predict = model.predict(X_valid)
# test_pred_proba = model.predict_proba(X_valid)
# #print('The MacroF1 of the train_20% is:',metrics.accuracy_score(Y_valid,test_predict))
# report = classification_report(Y_valid,test_predict)
# report_dict = classification_report(Y_valid, test_predict,output_dict=True)
#
# print(report)
# print("The MacroF1 of the valid is:{:.2f}%".format(MacroF1(report_dict) * 100))

# with open('output.txt', 'w') as file:
#     file.write("MacroF1:{:.2f}%".format(MacroF1(report_dict) * 100))

# model.save_model('catboost_model.cbm')
# import os
# model_size = os.path.getsize('catboost_model.cbm')
# model_size_kb = model_size / 1024
# print(f"The model size is {model_size_kb} kb.")
#

test_filename = r'D:\软件杯大赛\测试集\test_2000_x.csv'
#test = data_solve_RFECV_test(test_filename,100)
test  =data_processing_nomal_test(test_filename)
x_test = test.drop(['sample_id'],axis = 1)
test_predict = model.predict(x_test)
# print(type(test_predict))
print(type(test_predict))
print(test_predict.shape)
#test_pred_proba = model.predict_proba(x_valid)

pred_count = []
for i in range(6):
    pred_count.append(np.count_nonzero(test_predict == i))
    print("类别{}的数量：{}".format(i,pred_count[i]))
print("数据总数：{}".format(np.size(test_predict)))

np.savetxt("result.txt",test_predict,fmt='%d', delimiter=',')

# with open("result.txt", "w") as file:
#     json.dump(data, file)

print(test_predict)

data_list = test_predict.tolist()
data = {}
for i, value in enumerate(data_list):
    key = str(i)
    value = data_list[i][0]
    data[key] = value
# 将JSON字符串写入文件
with open("sumbit.json", "w") as file:
    json.dump(data, file)