import pandas as pd

from data_solve import data_processing_nomal


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)



train_filename = r"E:\杂项\软件杯大赛\训练数据集\preprocess_train.csv"
df = data_processing_nomal(train_filename)


# 假设df是你的DataFrame，'label'是你的目标标签
correlation_matrix = df.corr()
print(correlation_matrix['label'])
#19 - 0.16
#22 - 0.48
#67 - 0.1