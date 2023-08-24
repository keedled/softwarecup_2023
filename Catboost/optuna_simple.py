"""
Optuna example that optimizes a classifier configuration for cancer dataset using
Catboost.

In this example, we optimize the validation accuracy of cancer detection using
Catboost. We optimize both the choice of booster model and their hyperparameters.

"""

import numpy as np
import optuna

import catboost as cb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score,make_scorer
from sklearn.model_selection import train_test_split, cross_val_score

from data_solve import data_solve_RFECV
# Best trial:
#   Value: 0.8687104322721637
#   Params:
#     random_strength: 8
#     n_estimators: 787
#     colsample_bylevel: 0.7134197145457797
#     depth: 5
#     learning_rate: 0.2767789755159874
#     l2_leaf_reg: 3.0217858996548883
#     rangdom_seed: 65
#     boosting_type: Ordered
#     bootstrap_type: Bernoulli
#     subsample: 0.7793679490463288

# Best trial:
#   Value: 0.8676887911252852
#   Params:
#     random_strength: 4
#     n_estimators: 976
#     colsample_bylevel: 0.994827156205528
#     depth: 8
#     learning_rate: 0.45844206227094975
#     l2_leaf_reg: 3.3349186850844164
#     random_seed: 68
#     boosting_type: Plain
#     bootstrap_type: MVS
def custom_scoring(y_true, y_pred):#自定义参数评分方式,sklearn自动计算预测值。
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    macro_F1 = (2 * precision * recall) / (precision + recall)
    # 自定义评估指标的计算逻辑
    # 返回评估指标的值
    return macro_F1
custom_scorer = make_scorer(custom_scoring, greater_is_better=True)
data_filename = r"E:\杂项\软件杯大赛\训练数据集\preprocess_train.csv"
data = data_solve_RFECV(data_filename,61)
X = data.drop(['sample_id', 'label'], axis=1)
y = data[['label']]
def objective(trial):
    train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.225,shuffle=True,)

    param = {
        #'bagging_temperature': trial.suggest_int('bagging_temperature', 1, 10),
        'random_strength': trial.suggest_int('random_strength', 1, 10),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        #"objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 1),
        #colsample_bylevel 接受一个浮点数作为参数，表示每一级中要选择的特征列的比例。例如，如果 colsample_bylevel=0.5，则每一级将从所有特征列中随机选择一半的特征列用于训练。
        "depth": trial.suggest_int("depth", 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.01, 5),
        'random_seed': trial.suggest_int('random_seed', 1, 100),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        ),
        #"used_ram_limit": "4gb",
        #"used_ram_limit"：这个参数表示 CatBoost 在训练过程中使用的最大 RAM。在你的代码中，这个值被固定为 "3gb"。
        # 注意这个参数对于大型数据集和复杂模型是非常重要的，因为它们可能会消耗大量的内存。这个参数可以帮助你防止由于内存不足而导致的程序崩溃。
        # 如果你的设备有更多可用的内存，你可以考虑增加这个值以提高训练速度。同时，这个值也不能设置得过大，以免过度消耗内存，影响其他程序的运行。
    }
    #"bootstrap_type" 是 CatBoost 中用于指定 bootstrap 算法类型的参数。Bootstrap 是一种用于生成训练子集的采样技术，通常用于 bagging（自举汇聚）类型的集成方法。
    # 在 CatBoost 中，"bootstrap_type" 参数可以是 "Bayesian"、"Bernoulli" 或 "MVS"。
    #"Bayesian"：在这种模式下，样本被选中的概率是从贝叶斯分布中随机抽取的。此外，"bagging_temperature" 参数可以被用于控制分布的温度（更高的值导致更大的样本差异）。
    #"Bernoulli"：在这种模式下，每个样本被选中的概率都是独立且固定的。你可以使用 "subsample" 参数来指定这个固定的概率。
    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    model = cb.CatBoostClassifier(
        #auto_class_weights='Balanced',
        **param,
        verbose = 100,
    )

    model.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=100)

    preds = model.predict(valid_x)
    pred_labels = np.rint(preds)
    score = cross_val_score(model, X, y, cv=5, scoring=custom_scorer)
    return score.mean()


if __name__ == "__main__":# 只有在脚本直接执行时才会执行的代码块
    study = optuna.create_study(direction="maximize",pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=5,
                interval_steps=3
    #n_startup_trials: 前几轮试验不进行减枝。默认值是5。
    #n_warmup_steps: 在一个试验的前几步不进行减枝。默认值是0。
    #interval_steps: 每隔多少步进行一次减枝。默认值是1。
                                    ))
    study.optimize(objective, n_trials=30)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))