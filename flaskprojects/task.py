import json
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from flask import json, url_for
from app_config import  make_celery,create_app
from catboost import CatBoostClassifier,metrics
from data_solve import data_processing_nomal,data_solve_RFECV_test
from sklearn.metrics import classification_report
import pickle
import catboost
from database import db, Model, Prediction
import pandas as pd
app=create_app()
celery = make_celery(app)

def MacroF1(report):
    prec_avg = report['macro avg']['precision']
    reca_avg = report['macro avg']['recall']
    macro_F1 = (2 * prec_avg * reca_avg) / (prec_avg + reca_avg)
    return macro_F1
class ProgressCallback:
    def __init__(self, task_id, total_iterations):
        self.task_id = task_id
        self.total_iterations = total_iterations

    def after_iteration(self, info):
        iteration = info.iteration
        progress_percentage = iteration + 1
        app.redis_conn.set(f"train_progress_{self.task_id}", str(progress_percentage))
        return True

@celery.task(bind=True)
def train_model(self, file_path,model_name, user_id,dataset_name):
    logging.info(f'Starting train_model task with file_path={file_path}, model_name={model_name}, user_id={user_id}, dataset_name={dataset_name}')
    with app.app_context():
        start_time = datetime.now()
        dataset = pd.read_csv(file_path)
        model = CatBoostClassifier(auto_class_weights='Balanced',
                                   iterations = 100,
                              #     eval_metric = 'Accuracy')
                                  # train_dir = 'atboost_info',
                                 custom_metric=[metrics.TotalF1(), metrics.Accuracy(), metrics.MultiClass()])
        x_train = dataset.drop(['sample_id', 'label'], axis=1)
        y_train = dataset[['label']]
        train_pool = catboost.Pool(data=x_train, label=y_train)
        # def on_iteration_callback(iteration, total_iterations, task_id):
        #     # progress_percentage = (iteration + 1) / total_iterations * 100
        #     progress_percentage=iteration+1
        #     app.redis_conn.set(f"train_progress_{task_id}", str(progress_percentage))
        total_iterations = 100
        callback_instance = ProgressCallback(self.request.id, total_iterations)
        model.fit(train_pool,  callbacks=[callback_instance])

        # model.fit(train_pool)
        time = (datetime.now() - start_time).total_seconds()
        # 计算模型得分
        y_pred = model.predict(x_train)
        report = classification_report(y_train, y_pred, output_dict=True)
        score = MacroF1(report)
        new_model = Model(name=model_name, dataset_name=dataset_name, score=score, total_time=time, user_id=user_id)
        db.session.add(new_model)
        db.session.flush()
        model_name_with_id = f"{new_model.id}_{model_name}"
        file_path = f'models/{model_name_with_id}.pkl'
        pickle.dump(model, open(file_path, 'wb'))
        new_model.name = model_name_with_id
        new_model.file_path = file_path
        db.session.commit()

        eval_results = model.get_evals_result()
        # 获取训练过程中的指标
        train_acc = eval_results['learn']['Accuracy:use_weights=true']
        train_mul = eval_results['learn']['MultiClass:use_weights=false']
        train_F1 = eval_results['learn']['TotalF1:use_weights=false']
        model_id = new_model.id
        url1="train_img/total/img1_"+str(model_id)+".png"
        url2="train_img/accuracy/img2_"+str(model_id)+".png"
        url3="train_img/mult/img3_" + str(model_id) + ".png"
        # y_train_pred = model.predict(train_x)
        # train_accuracy = accuracy_score(train_y, y_train_pred)
        # train_log_loss = log_loss(train_y, y_train_pred)
        # ------------可视化准确率------------------
        iterations1 = range(len(train_acc))
        plt.figure()
        plt.plot(iterations1, train_acc, label='Train Accuracy')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy')
        plt.title('CatBoost Training Accuracy')
        plt.legend()
        plt.savefig(url2)
        plt.close()
        # ------------可视化损失指------------------
        iterations2 = range(len(train_mul))
        plt.figure()
        plt.plot(iterations2, train_mul, label='Train MultiClass')
        plt.xlabel('Iterations')
        plt.ylabel('MultiClass')
        plt.title('CatBoost Training MultiClass')
        plt.legend()
        plt.savefig(url3)
        plt.close()
        # ------------可视化F1------------------
        iterations3 = range(len(train_F1))
        plt.figure()
        plt.plot(iterations3, train_F1, label='Train F1')
        plt.xlabel('Iterations')
        plt.ylabel('F1')
        plt.title('CatBoost Training F1')
        plt.legend()
        plt.savefig(url1)
        plt.close()

        new_model.url1=url1
        new_model.url2=url2
        new_model.url3=url3
        db.session.commit()
        logging.info("yes")
        return {'message': 'Model trained and saved aaaaa'}

@celery.task(bind=True)
def test_model(self, file_path, model_id, user_id):
    with app.app_context():
        model_record = Model.query.get(model_id)
        dataset = pd.read_csv(file_path)
        if model_record is None:
            return {'message': 'Model not found'}
        if model_record.user_id != user_id:  # 验证用户
            return {'message': 'Permission denied'}
        try:
            model = pickle.load(open(model_record.file_path, 'rb'))
        except FileNotFoundError:
            return {'message': 'Model file not found'}
        x_test = dataset.drop(['sample_id'], axis=1)
        test_predict = model.predict(x_test)
        # print(type(test_predict))
        print(test_predict)
        # test_pred_proba = model.predict_proba(x_valid)
        pred_count = []
        for i in range(6):
            pred_count.append(np.count_nonzero(test_predict == i))
            print("类别{}的数量：{}".format(i, pred_count[i]))
        print("数据总数：{}".format(np.size(test_predict)))

        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False

        label_list = ["类别0", "类别1", "类别2", "类别3", "类别4", "类别5"]  # 各部分标签
        size = pred_count  # 各部分大小
        color = ["#4D49BE", "#486AFF", "#00BC7B", "#F9423A", "#F6A04D", "#F3D321"]  # 各部分颜色
        # explode = [0.05, 0, 0]   # 各部分突出值
        """
        绘制饼图
        explode：设置各部分突出
        label:设置各部分标签，
        labeldistance:设置标签文本距圆心位置，1.1表示1.1倍半径
        autopct：设置圆里面文本
        shadow：设置是否有阴影
        startangle：起始角度，默认从0开始逆时针转
        pctdistance：设置圆内文本距圆心距离
        返回值
        l_text：圆内部文本，matplotlib.text.Text object
        p_text：圆外部文本
        """
        new_prediction = Prediction(
            class0_count=pred_count[0],
            class1_count=pred_count[1],
            class2_count=pred_count[2],
            class3_count=pred_count[3],
            class4_count=pred_count[4],
            class5_count=pred_count[5],
            user_id=user_id
        )
        db.session.add(new_prediction)
        db.session.commit()
        result_image_url = "results/img/result_" + str(new_prediction.id) + ".png"
        plt.figure()
        patches, l_text, p_text = plt.pie(size,
                                          colors=color,
                                          labels=label_list,
                                          labeldistance=1.1,
                                          autopct="%1.1f%%",
                                          # shadow=True,
                                          startangle=90,
                                          pctdistance=0.6)
        plt.axis("equal")  # 设置横轴和纵轴大小相等，这样饼才是圆的
        plt.legend()
        plt.savefig(result_image_url)
        plt.close()
        data_list = test_predict.tolist()
        data = {}
        for i, value in enumerate(data_list):
            key = str(i)
            value = data_list[i][0]
            data[key] = value
        with open("results/json/" + str(new_prediction.id) + ".json", "w") as file:
            json.dump(data, file)
        new_prediction.result_json_url = "results/json/" + str(new_prediction.id) + ".json"
        new_prediction.result_image_url = result_image_url
        db.session.commit()



