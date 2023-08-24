import numpy as np
from sklearn import metrics
import pandas as pd
from sklearn.metrics import classification_report
from data_solve import data_solve_train_valid,data_processing_nomal,data_solve_RFECV
from autogluon.tabular import TabularPredictor
#AutoGluon TabularPredictor 预测表格数据集（分类或回归）列中的值。
from custom_score import MacroF1
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
# y_true = np.random.randint(low=0, high=2, size=10)
# y_pred = np.random.randint(low=0, high=2, size=10)
# pd.set_option('display.max_columns', None)  # 显示所有列
# pd.set_option('display.max_rows', None)     # 显示所有行

# sklearn.metrics.accuracy_score(y_true, y_pred)
# print(sklearn.metrics.f1_score(y_true,y_pred))

from autogluon.core.metrics import make_scorer

def MacroF1(report):
    prec_avg = report['macro avg']['precision']
    reca_avg = report['macro avg']['recall']
    macro_F1 = (2 * prec_avg * reca_avg) / (prec_avg + reca_avg)
    return macro_F1

ag_f1_score_scorer = make_scorer(name='f1_score',
                                #'f1_macro': 宏平均F1值。
                                 score_func=f1_score,
                                 optimum=1,
                                 average='macro',
                                 greater_is_better=True)

train_filename = r"E:\杂项\软件杯大赛\训练数据集\preprocess_train.csv"
# train = data_processing_nomal(train_filename)
# train = train.drop(['sample_id'],axis = 1)
# valid_name = r"E:\杂项\软件杯大赛\验证集\validate_1000.csv"
# valid = data_processing_nomal(valid_name)
# valid = valid.drop(['sample_id'],axis = 1)
#
# X_train = train.drop(['label'], axis=1)
# Y_train = train[['label']]
# X_test = valid.drop(['label'],axis = 1)
# Y_test = valid[['label']]
# y_test = Y_test['label']


train_filename = r"E:\杂项\软件杯大赛\训练数据集\preprocess_train.csv"
train = data_solve_train_valid(train_filename)
x_train = train.drop(['sample_id', 'label'], axis=1)
y_train = train[['label']]

#print(train.head())


# train = data_solve_train_valid(train_filename)
# X = train.drop(['sample_id','label'],axis=1)
# Y = train[['label']]
# y = Y['label']
#x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.225, random_state=100,shuffle=True,stratify=Y)
Train = train.drop(['sample_id'],axis = 1)
#Train = pd.concat([x_train, y_train], axis=1)
# Valid = pd.concat([x_train,y_valid],axis=1)
#stratify=Y表示根据标签Y的分布进行分层抽样，确保训练集和验证集中各类别样本的比例与原始数据集中的比例相同。这在处理不平衡数据集时特别有用。
#save_path = 'AutogluonModels/medium_quality'
#time_limit = 72000
save_path = 'AutogluonModels/best_quality'
predictor = TabularPredictor(label='label',path = save_path,eval_metric=ag_f1_score_scorer,problem_type='multiclass').fit(
    Train,
    hyperparameters = 'light',
    auto_stack=True,
    #time_limit=time_limit,
    presets='best_quality',
    refit_full = True,#是否在训练完成后使用完整的训练数据重新训练最佳模型。
    set_best_to_refit_full = True,#是否将最佳模型设置为完整训练的模型。
    _save_bag_folds = True,#_save_bag_folds: 是否保存每个模型的袋外预测结果。
    keep_only_best = True,#keep_only_best: 是否只保留最佳模型，而不保留其他训练的模型。
    save_space = True,#save_space: 是否尽量减少保存模型所需的磁盘空间。
)
# predictor = TabularPredictor.load("AutogluonModels/best_quality")
# print(train.head())
# importance = predictor.feature_importance(train)
#print(importance)
# y_pred = predictor.predict(x_valid)
# report_dict = classification_report(y_valid, y_pred,output_dict=True)
# print("Predictions:  \n", y_pred)
# # perf = predictor.evaluate_predictions(y_true=y_valid, y_pred=y_pred,auxiliary_metrics=True) #如果你的主要预测指标是准确度（accuracy），
# #                                                                                             #但你也想了解模型的精确度和召回率，
# #                                                                                             #那么将auxiliary_metrics设置为True将计算并返回这些指标的值。
# print("MacroF1:{}".format(MacroF1(report_dict)))
# print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(y_valid,y_pred))
# report = classification_report(y_valid,y_pred)
# print(report)


test_filename = r'D:\软件杯大赛\测试集\test_2000_x.csv'
test = data_processing_nomal(test_filename)
x_test = test.drop(['sample_id'],axis = 1)
test_predict = predictor.predict(x_test)
# print(type(test_predict))
print(test_predict)
#test_pred_proba = model.predict_proba(x_valid)
count = []
for i in range(6):
    count.append(np.count_nonzero(test_predict == i))
    print("类别{}的数量：{}".format(i,count[i]))
print("数据总数：{}".format(np.size(test_predict)))
