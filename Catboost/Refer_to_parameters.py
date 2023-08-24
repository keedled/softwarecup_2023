import catboost
from catboost import CatBoostClassifier
from data_solve import data_processing_nomal
#from Predict import MacroF1
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, precision_score, recall_score
import numpy as np

def custom_scoring(y_true, y_pred):#自定义参数评分方式,sklearn自动计算预测值。
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    macro_F1 = (2 * precision * recall) / (precision + recall)
    # 自定义评估指标的计算逻辑
    # 返回评估指标的值
    return macro_F1


def MacroF1(report):
    prec_avg = report['macro avg']['precision']
    reca_avg = report['macro avg']['recall']
    macro_F1 = (2 * prec_avg * reca_avg) / (prec_avg + reca_avg)
    return macro_F1

train_filename = r"/训练数据集/preprocess_train.csv"
train = data_processing_nomal(train_filename)
valid_filename = '/验证集/validate_1000.csv'
valid = data_processing_nomal(valid_filename)
x_train = train.drop(['sample_id', 'label'], axis=1)
y_train = train[['label']]
x_valid = valid.drop(['sample_id','label'],axis = 1)
y_valid = valid[['label']]
train_pool = catboost.Pool(x_train,y_train)
valid_pool = catboost.Pool(x_valid,y_valid)

#custom_scorer = make_scorer(custom_scoring, greater_is_better=True)
#params_name:border_count border_count:52 Custom Score: 0.9027363818454771
#params_name:n_estimators learning_times:147 Custom Score: 0.8868
#params_name:border_count border_count:253 Custom Score: 0.9030018413740943
#params_name:random_seed random_seed:119 Custom Score: 0.8697535304980677
#params_name:random_seed random_seed:576 Custom Score: 0.8717817398244873
#params_name:random_seed random_seed:645 Custom Score: 0.8729463465711748
#params_name:L2_leaf_reg l2_leaf_reg:2.7777777777777777 Custom Score: 0.8822010565692298
#params_name:bagging_temperature bagging_temperature:146 Custom Score: 0.9030018413740943
def Mesh_param(params_name,iter):
    #len = np.linspace(2.5,3.5,iter)
    class_weights = {0:1,  1:4.908, 2:3.19, 3:5.86, 4:9.12, 5:6.90}
    for i in range(200,iter):
        model = CatBoostClassifier(
            bagging_temperature = 1,
            random_strength = 1,
            n_estimators = 146,
            learning_rate=0.094,
            depth=6,
            l2_leaf_reg=3.459,
            # loss_function='MultiClass',
            # eval_metric='MultiClass',
            random_seed=645,
            #early_stopping_rounds=None,
            #use_best_model=True,
            verbose=0,
            per_float_feature_quantization='0:border_count={}'.format(i),


            class_weights = class_weights,
            #max_leaves
            #min_data_in_leaf

            #ignored_features
        )
        #model_stacking = CatBoostClassifier(n_estimators=i)
        model.fit(train_pool)
        y_pred_valid = model.predict(x_valid)
        report_dict = classification_report(y_valid, y_pred_valid, output_dict=True)
        score = MacroF1(report_dict)
        print("params_name:{}".format(params_name),"border_count:{}".format(i),"Custom Score: {}".format(score))



#param =
Mesh_param("border_count",300)



