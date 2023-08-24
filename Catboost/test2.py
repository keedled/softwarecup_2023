from sklearn.feature_selection import mutual_info_classif as MIC
from data_solve import data_solve_train_valid,data_processing_nomal,data_processing_nomal_test
import pandas as pd
import torch

x = torch.randn(10, 15, 30)
print(x.shape)  # 输出: torch.Size([10, 1, 30])

y = x.squeeze(-1)
print(y.shape)  # 输出仍为: torch.Size([10, 1, 30]) 因为第-1维（即第2维）的大小不是1

# y = x.squeeze(1)
# print(y.shape)  # 输出: torch.Size([10, 30])，因为第1维的大小为1，被移除了
