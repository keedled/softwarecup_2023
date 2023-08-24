import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.callbacks import VerboseCallback
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, precision_score, recall_score
from data_solve import data_processing_nomal,data_solve_train_valid
import catboost

def custom_scoring(estimator, X, y_true):#自定义参数评分方式,sklearn自动计算预测值。
    y_pred = estimator.predict(X)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    macro_F1 = (2 * precision * recall) / (precision + recall)
    # 自定义评估指标的计算逻辑
    # 返回评估指标的值
    return macro_F1

def on_step(optim_result):
    score = opt.best_score_
    print("Best score: %s" % score)
    if score >= 0.98:
        print('Interrupting!')
        return True

def MacroF1(report):
    prec_avg = report['macro avg']['precision']
    reca_avg = report['macro avg']['recall']
    macro_F1 = (2 * prec_avg * reca_avg) / (prec_avg + reca_avg)
    return macro_F1

data_filename = r"E:\杂项\软件杯大赛\训练数据集\preprocess_train.csv"
data = data_solve_train_valid(data_filename)
X = data.drop(['sample_id', 'label'], axis=1)
y = data[['label']]


x_train, x_valid, y_train, y_valid = train_test_split(data, y, test_size=0.2, random_state=42,shuffle=True,stratify=y)
train_pool = catboost.Pool(x_train,y_train)
#valid_pool = catboost.Pool(x_valid,y_valid)

class_weights = {0:1,  1:4.908, 2:3.19, 3:5.86, 4:9.12, 5:6.90}
search_spaces = {
            # 'bagging_temperature' : Integer(1,10,'uniform'),
            # 'random_strength' : Integer(1,10,'uniform'),
            # 'n_estimators' : Integer(1,1000,'uniform'),
            #'learning_rate' : Real(0.01,1,'log-uniform'),
            # 'depth' : Integer(3,8,'uniform'),
            #'l2_leaf_reg' : Real(0.01,5,'log-uniform'),
            # 'random_seed' : Integer(1,100,'uniform'),
            'loss_function':['MultiClass','MultiClassOneVsAll']
}

catboost = CatBoostClassifier(
                            #class_weights = class_weights,
                            auto_class_weights='Balanced',
                            #boosting_type = 'Ordered',
                            # bagging_temperature = 2,
                            # random_strength = 155,
                            # depth = 5,
                            # random_seed = 1255,
                            # n_estimators = 492,
    )
#X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
opt = BayesSearchCV(catboost, search_spaces, scoring=custom_scoring, cv=5, n_iter=5)
callback = VerboseCallback(10)
opt.fit(X, y,callback = on_step)


# 打印最佳参数
# with open('output.txt', 'w') as file:
#     file.write("Best parameters found: ")
#     file.write(str(opt.best_params_))
#     file.write("Best accuracy found: " + str(opt.score(x_valid, y_valid)))
print("Best parameters found: ", opt.best_params_)

# # 验证模型
# print("Best accuracy found: ", opt.score(x_valid, y_valid))