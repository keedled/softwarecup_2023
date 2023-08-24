from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
from sklearn.preprocessing import StandardScaler
import catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from data_solve import data_processing_nomal
#from Predict import MacroF1
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, precision_score, recall_score


def data_processing_iter(path):
    df = pd.read_csv(path)
    features = df.iloc[:, 1:-1]
    imp = IterativeImputer(max_iter=50, random_state=0)
    features = pd.DataFrame(imp.fit_transform(features))
    #print(df)
    scaler = StandardScaler()
    # print(type(features))  <class 'pandas.core.frame.DataFrame'>
    scaler.fit(features)  # fit方法用于计算特征的均值和标准差，这些值将用于后续的标准化（缩放）过程。
    features = scaler.transform(features)  # transform方法可以正确地对数据进行标准化。
    # print(type(features))   <class 'pandas.core.frame.DataFrame'>
    features = pd.DataFrame(features)
    #numeric_features = features.dtypes[features.dtypes != 'object'].index  # 获得数值型列索引
    # features[numeric_features] = features[numeric_features].fillna(0)  # 用0填充数值型列的空值
    features_labels = pd.concat([features, df[['label']]], axis=1)
    df = pd.concat([df[['sample_id']], features_labels], axis=1)
    return df

train = data_processing_iter('E:\杂项\软件杯大赛\训练数据集\preprocess_train.csv')
test = data_processing_iter('E:\\杂项\\软件杯大赛\\验证集\\validate_1000.csv')
# print(df.describe())
# print(df.info())

def MacroF1(report):
    prec_avg = report['macro avg']['precision']
    reca_avg = report['macro avg']['recall']
    macro_F1 = (2 * prec_avg * reca_avg) / (prec_avg + reca_avg)
    return macro_F1


model = CatBoostClassifier()

x_train = train.drop(['sample_id', 'label'], axis=1)
y_train = train[['label']]

X_test = test.drop(['sample_id','label'],axis = 1)
Y_test = test[['label']]
train_pool = catboost.Pool(x_train,y_train)
model.fit(train_pool)

test_predict = model.predict(X_test)

test_pred_proba = model.predict_proba(X_test)
#print(y_pred_proba)

print('The accuracy of the Logistic Regression is:',metrics.accuracy_score(Y_test,test_predict))
report = classification_report(Y_test,test_predict)
report_dict = classification_report(Y_test, test_predict,output_dict=True)

print(report)
print("MacroF1:{:.2f}%".format(MacroF1(report_dict) * 100))