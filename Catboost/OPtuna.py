import optuna
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, precision_score, recall_score
from data_solve import data_processing_nomal,data_solve_train_valid,data_solve_RFECV
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
import catboost
from sklearn.model_selection import cross_val_score
import optuna.visualization as ov
def custom_scoring(y_true, y_pred):#自定义参数评分方式,sklearn自动计算预测值。
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    macro_F1 = (2 * precision * recall) / (precision + recall)
    # 自定义评估指标的计算逻辑
    # 返回评估指标的值
    return macro_F1
custom_scorer = make_scorer(custom_scoring, greater_is_better=True)
# Best trial:
#   Value: 0.8613741422632444
#   Params:
#     learning_rate: 0.062013233797374154
#     depth: 6
# Best trial:
#   Value: 0.8654123984928604
#   Params:
#     bagging_temperature: 1
#     random_strength: 1
#     n_estimators: 711
#     l2_leaf_reg: 0.2043312644520966
#     rangdom_seed: 77


data_filename = r"E:\杂项\软件杯大赛\训练数据集\preprocess_train.csv"
data = data_solve_RFECV(data_filename,61)
X = data.drop(['sample_id', 'label'], axis=1)
y = data[['label']]
x_train, x_valid, y_train, y_valid = train_test_split(data, y, test_size=0.2, random_state=42,shuffle=True,stratify=y)
train_pool = catboost.Pool(x_train,y_train)
print(X.describe())
def objective(trial):
    params = {
        'bagging_temperature' : trial.suggest_int('bagging_temperature',1,10),
        'random_strength' : trial.suggest_int('random_strength',1,10),
        'n_estimators' : trial.suggest_int('n_estimators',1,1000),
        'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 1),
        'depth' : trial.suggest_int('depth', 3, 8),
        'l2_leaf_reg' : trial.suggest_loguniform('l2_leaf_reg',0.01,5),
        'random_seed' : trial.suggest_int('rangdom_seed',1,100),
        #'loss_function': trial.suggest_categorical('loss_function',['MultiClass', 'MultiClassOneVsAll'])
    }
    model = CatBoostClassifier(
        **params,
        #depth = 6,
        #learning_rate = 0.0620132338,
        loss_function  = 'MultiClass',
        #class_weights = class_weights,
        #auto_class_weights='Balanced',
        boosting_type = 'Ordered',
        verbose=100
    )
    score = cross_val_score(model, X, y, cv=5,scoring=custom_scorer)
    return score.mean()
study = optuna.create_study(direction='maximize', study_name='example-study', load_if_exists=True,
                            pruner=optuna.pruners.MedianPruner(
                            n_startup_trials=10,
                            n_warmup_steps=5,
                            interval_steps=3
                            ))
# study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(
#                 n_startup_trials=10,
#                 n_warmup_steps=5,
#                 interval_steps=3
#     #n_startup_trials: 前几轮试验不进行减枝。默认值是5。
#     #n_warmup_steps: 在一个试验的前几步不进行减枝。默认值是0。
#     #interval_steps: 每隔多少步进行一次减枝。默认值是1。
#                                     ))
study.optimize(objective, n_trials=30,timeout=450)
study.trials_dataframe().to_csv("study_history.csv")
print("Number of finished trials: {}".format(len(study.trials)))
#study.trials：这是一个包含了所有试验的列表。每个试验都是一个Trial对象，包含了这次试验的所有信息，例如参数、目标函数值等。
print("Best trial:")
#study.best_trial：这表示目标函数值最好的试验。
trial = study.best_trial

print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))



