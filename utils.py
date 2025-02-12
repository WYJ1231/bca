import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from scipy import stats

save_dir = '/statistical/visualization/'

import os
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

csv_file_path_1 = '/experiment_AxR_diceAndBce_loss_test/ind_seg_result.csv'
csv_file_path_2 = '/statistical/unet_cross_3/AxR/metrics_mul.csv'

df1 = pd.read_csv(csv_file_path_1)
df2 = pd.read_csv(csv_file_path_2)

metric = ["Dice", "Hausdorff Distance", "Average Surface Distance"]

model_a_dice = df1["AxR_ASD"].values
model_b_dice = df2[:-2][metric].values

t_stat, p_value = ttest_rel(model_a_dice, model_b_dice)

print("t-statistic:", t_stat)
print("p-value:", p_value)

alpha = 0.05
if p_value < alpha:
    print(f"两个模型在{metric}系数上的表现有显著差异。")
else:
    print(f"两个模型在{metric}系数上的表现没有显著差异。")
