import catboost
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from data_solve import data_processing_nomal, data_solve_train_valid, data_solve_train_valid, data_solve_RFECV_2, \
    data_solve_RFECV_test
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
                            n_estimators = 500,
                            # learning_rate = 0.12954489804080596,
                            # random_seed = 1255,
                            # l2_leaf_reg = 0.4,
                            # # depth=5,
                            # # bagging_temperature = 2,
                            # # random_strength = 155,
                            # #per_float_feature_quantization=['10:border_count=1024','22:border_count=1024'],
                            # #per_float_feature_quantization=['10:border_count=1024'],
                            #auto_class_weights='Balanced',#MacroF1:84.63%
                            # #class_weights = class_weights,
                            # boosting_type = 'Ordered'
                            #approx_on_full_history = True,
                            #最后两个参数可以后续测试。
)

# data_filename = r"E:\杂项\软件杯大赛\训练数据集\preprocess_train.csv"
# data = data_solve_train_valid(data_filename)

train_filename = r"E:\杂项\软件杯大赛\训练数据集\preprocess_train.csv"
train = data_processing_nomal(train_filename)
valid_filename = 'E:\\杂项\\软件杯大赛\\验证集\\validate_1000.csv'
valid = data_processing_nomal(valid_filename)

X_train = train.drop(['sample_id', 'label'], axis=1)
Y_train = train[['label']]

X_valid = valid.drop(['sample_id', 'label'], axis=1)
Y_valid = valid[['label']]

# X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=100)
train_pool = catboost.Pool(X_train,Y_train)
model.fit(train_pool,verbose = 100)

# 获取特征重要性
feature_importances = model.get_feature_importance()

# 打印特征重要性
for score, name in sorted(zip(feature_importances, X_train.columns), reverse=True):
    print('{}: {}'.format(name, score))

important_features_mask = feature_importances > 0

# 使用这个布尔数组来索引特征名称，得到所有重要性大于0的特征
features_matrix_Var_importance = X_train.columns[important_features_mask]
x_train = X_train.filter(items = features_matrix_Var_importance)

print(x_train.shape)
