# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 18:06:06 2024

@author: Liu Yang

@project:
"""
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sklearn.linear_model import LinearRegression

path0 = Path("batch_sum.pkl")
batch0 = pickle.load(open(path0, 'rb'))

num_bat = len(batch0.keys())

features_df = pd.DataFrame()
features_df['cell_key'] = np.array(list(batch0.keys()))
features_df.head()

#特征dQ100-Q10
minimum_dQ_100_10 = np.zeros(num_bat)
variance_dQ_100_10 = np.zeros(num_bat)
skewness_dQ_100_10 = np.zeros(num_bat)
kurtosis_dQ_100_10 = np.zeros(num_bat)

for i, cell in enumerate(batch0.values()):
    c10 = cell['cycles']['10']
    c100 = cell['cycles']['100']
    dQ_100_10 = c100['Qdlin'] -c10['Qdlin']
    
    
    minimum_dQ_100_10[i] = np.log(np.abs(np.min(dQ_100_10)))
    variance_dQ_100_10[i] = np.log(np.var(dQ_100_10))
    skewness_dQ_100_10[i] = np.log(np.abs(skew(dQ_100_10)))
    kurtosis_dQ_100_10[i] = np.log(np.abs(kurtosis(dQ_100_10)))


features_df["minimum_dQ_100_10"] = minimum_dQ_100_10
features_df["variance_dQ_100_10"] = variance_dQ_100_10
features_df["skewness_dQ_100_10"] = skewness_dQ_100_10
features_df["kurtosis_dQ_100_10"] = kurtosis_dQ_100_10

features_df.head()


#放电容量曲线特征

slope_lin_fit_2_100 = np.zeros(num_bat) # 2-100圈斜率
intercept_lin_fit_2_100 = np.zeros(num_bat)
discharge_capacity_2 = np.zeros(num_bat)
diff_discharge_capacity_max_2 = np.zeros(num_bat)

for i, cell in enumerate(batch0.values()):
    #计算2至100圈线性拟合
    q = cell['summary']['QD'][1:100].reshape(-1,1) # 放电容量形状(99,1)
    X = cycle_numbers = cell['summary']['cycle'][1:100].reshape(-1,1)
    
    linear_regressor_2_100 = LinearRegression()
    linear_regressor_2_100.fit(X,q)
    
    slope_lin_fit_2_100[i] = linear_regressor_2_100.coef_[0]
    intercept_lin_fit_2_100[i] = linear_regressor_2_100.intercept_
    discharge_capacity_2[i] = q[0][0]
    diff_discharge_capacity_max_2[i] = np.max(q) - q[0][0]


features_df["slope_lin_fit_2_100"] = slope_lin_fit_2_100
features_df["intercept_lin_fit_2_100"] = intercept_lin_fit_2_100
features_df["discharge_capacity_2"] = discharge_capacity_2
features_df["diff_discharge_capacity_max_2"] = diff_discharge_capacity_max_2 



# 其他特征

mean_charge_time_2_6 = np.zeros(num_bat)
minimun_IR = np.zeros(num_bat)
diff_IR_100_2 = np.zeros(num_bat)
integral_tem_2_100 = np.zeros(num_bat)


for i, cell in enumerate(batch0.values()):
    mean_charge_time_2_6[i] = np.mean(cell['summary']['chargetime'][1:6]) #前五圈平均充电时间
    minimun_IR[i] = np.min(cell['summary']['IR'][1:100])
    diff_IR_100_2[i] = cell['summary']['IR'][100] - cell['summary']['IR'][1]
    
    integral_0 = 0
    for j in range(1,101):
        xx = cell["cycles"][str(j)]["t"]
        yy = cell["cycles"][str(j)]["T"]
        integral_0 = np.trapz(yy,xx) + integral_0
    
    integral_tem_2_100[i] = integral_0 # 2-100圈温度-时间积分
    
features_df["mean_charge_time_2_6"] = mean_charge_time_2_6
features_df["minimun_IR"] = minimun_IR
features_df["diff_IR_100_2"] = diff_IR_100_2
features_df["integral_tem_2_100"] = integral_tem_2_100

features_df.head()



# 其他分类特征与目标
# Classifier features
minimum_dQ_5_4 = np.zeros(num_bat)
variance_dQ_5_4 = np.zeros(num_bat)
cycle_550_clf = np.zeros(num_bat)
target_mod = np.zeros(num_bat)

for i, cell in enumerate(batch0.values()):
    c4 = cell['cycles']['4']
    c5 = cell['cycles']['5']
    dQ_5_4 = c5['Qdlin'] - c4['Qdlin']
    minimum_dQ_5_4[i] = np.log(np.abs(np.min(dQ_5_4)))
    variance_dQ_5_4[i] = np.log(np.var(dQ_5_4))
    cycle_550_clf[i] = cell['cycle_life'] >= 550
    
    #目标
    target_mod[i] = cell['cycle_life']

features_df["minimum_dQ_5_4"] = minimum_dQ_5_4
features_df["variance_dQ_5_4"] = variance_dQ_5_4
features_df["cycle_550_clf"] = cycle_550_clf
features_df["cycle_life"] = target_mod




with open('rebuild_features.pkl','wb') as fp:
        pickle.dump(features_df,fp)

features_df.to_csv('rebuild_features.csv', index=False)


# plt.plot(batch0["b1c0"]["cycles"]["1054"]["t"], batch0["b1c0"]["cycles"]["1054"]["T"])
# plt.grid()
# plt.show()


# plt.plot(batch0["b1c0"]["cycles"]["1054"]["Tdlin"])
# plt.grid()
# plt.show()

    
print(features_df.keys())










