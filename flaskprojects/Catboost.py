import catboost
from catboost import CatBoostClassifier
from data_solve import data_processing_nomal,data_solve_train_valid
#from Predict import MacroF1
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
def MacroF1(report):
    prec_avg = report['macro avg']['precision']
    reca_avg = report['macro avg']['recall']
    macro_F1 = (2 * prec_avg * reca_avg) / (prec_avg + reca_avg)
    return macro_F1

train_filename = r"E:\flaskprojects\preprocess_train.csv"
train = data_solve_train_valid(train_filename)
test_filename = 'E:\\flaskprojects\\validate_1000.csv'
test = data_processing_nomal(test_filename)

class_weights = {0:1,  1:4.908, 2:3.19, 3:5.86, 4:9.12, 5:6.90}#设置类的权重，一般来说类别较少的类应该有更高的权重。
model = CatBoostClassifier(
                            # n_estimators = 492,
                            # learning_rate = 0.12954489804080596,
                            # random_seed = 1255,
                            # l2_leaf_reg = 0.3028,
                            # depth=5,
                            # bagging_temperature = 2,
                            # random_strength = 155,
                            #per_float_feature_quantization=['10:border_count=1024','22:border_count=1024'],
                            #per_float_feature_quantization=['10:border_count=1024'],
                            #auto_class_weights='Balanced',#MacroF1:84.63%
                            #class_weights = class_weights,
                            #boosting_type = 'Ordered'
                            #3train_dir = 'cat_info'
                            custom_metric=[metrics.TotalF1(),metrics.Accuracy(),metrics.MultiClass]
                            #approx_on_full_history = True,
                            #最后两个参数可以后续测试。
)

x_train = train.drop(['sample_id', 'label'], axis=1)
y_train = train[['label']]

X_test = test.drop(['sample_id','label'],axis = 1)
Y_test = test[['label']]

#X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=100)
train_pool = catboost.Pool(x_train,y_train)
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
model.fit(train_pool)
# y_pred = model.predict(test_x)
# report = classification_report(test_y,y_pred)
# report_dict = classification_report(test_y, y_pred,output_dict=True)
# # f1_average = custom_f1_average(test_x, y_pred)
# print(report)
# print(report_dict)
# 获取训练过程中的最佳指标值
#evals_result = model.get_evals_result()

# for eval_set_name, eval_set_history in evals_result.items():
#     print(f"Evaluation set: {eval_set_name}")
#     for metric_name, metric_values in eval_set_history.items():
#         print(f"{metric_name}: {metric_values}")
#     print()
evals_result = model.get_evals_result()
train_TotalF1 = evals_result['learn']['TotalF1']
train_accuracy = evals_result['learn']['Accuracy']
train_MultiClass = evals_result['learn']['MultiClass']
#validation_accuracy = evals_result['validation']['TotalF1']

# 绘制准确率的图表
plt.plot(train_TotalF1, label='Train TotalF1')
#plt.plot(train_accuracy, label='Train Accuracy')
#plt.plot(train_Logloss, label='Train Logloss')
#plt.plot(validation_accuracy, label='Validation TotalF1')
plt.xlabel('Iteration')
plt.ylabel('TotalF1')
plt.legend()

plt.figure()  # 创建新的图形窗口
plt.plot(train_MultiClass, label='Train MultiClass')
plt.xlabel('Iteration')
plt.ylabel('MultiClass')
plt.legend()

plt.figure()  # 创建新的图形窗口
plt.plot(train_accuracy, label='Train Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuarcy')
plt.legend()

plt.show()
