# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:45:14 2024

@author: Liu Yang

@project:
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

batch1 = pickle.load(open(r'E:\\inc\\code\\soh\\a_my_codes\\nature_energy\\datasets\\batch1.pkl', 'rb'))
#remove batteries that do not reach 80% capacity
del batch1['b1c8']
del batch1['b1c10']
del batch1['b1c12']
del batch1['b1c13']
del batch1['b1c22']

numBat1 = len(batch1.keys())
print(numBat1)
#########################################################
batch2 = pickle.load(open(r'E:\\inc\\code\\soh\\a_my_codes\\nature_energy\\datasets\\batch2.pkl', 'rb'))
# There are four cells from batch1 that carried into batch2, we'll remove the data from batch2
# and put it with the correct cell from batch1
batch2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
add_len = [662, 981, 1060, 208, 482];

for i, bk in enumerate(batch1_keys):
    batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + add_len[i]
    for j in batch1[bk]['summary'].keys():
        if j == 'cycle':
            batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j] + len(batch1[bk]['summary'][j])))
        else:
            batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j]))
    last_cycle = len(batch1[bk]['cycles'].keys())
    for j, jk in enumerate(batch2[batch2_keys[i]]['cycles'].keys()):
        batch1[bk]['cycles'][str(last_cycle + j)] = batch2[batch2_keys[i]]['cycles'][jk]

del batch2['b2c7']
del batch2['b2c8']
del batch2['b2c9']
del batch2['b2c15']
del batch2['b2c16']

numBat2 = len(batch2.keys())
print(numBat2)

#########################################################
batch3 = pickle.load(open(r'E:\\inc\\code\\soh\\a_my_codes\\nature_energy\\datasets\\batch3.pkl', 'rb'))
# remove noisy channels from batch3
del batch3['b3c37']
del batch3['b3c2']
del batch3['b3c23']
del batch3['b3c32']
del batch3['b3c42']
del batch3['b3c43']

numBat3 = len(batch3.keys())
print(numBat3)

numBat = numBat1 + numBat2 + numBat3
numBat

# 训练集、测试集划分
test_ind = np.hstack((np.arange(0,(numBat1+numBat2),2),83))
train_ind = np.arange(1,(numBat1+numBat2-1),2)
secondary_test_ind = np.arange(numBat-numBat3,numBat)


####合并所有batch
bat_dict = {**batch1, **batch2, **batch3}

# 保存数据
with open('batch_sum.pkl','wb') as fp:
        pickle.dump(bat_dict,fp)

data_split = {'test':test_ind, 'train':train_ind, 'valid':secondary_test_ind}
with open('data_split.pkl','wb') as fp:
        pickle.dump(data_split,fp)

###################
cycle_life = []
x_group = []
y_group = []
color_m = []
for i in bat_dict.keys():
    cycle_life.append(bat_dict[i]['cycle_life'])    

cyc_max = np.array(cycle_life).max()
####################

for i in bat_dict.keys():
    num = len(bat_dict[i]['summary']['cycle'])
    x_group.append(bat_dict[i]['summary']['cycle'])
    y_group.append(bat_dict[i]['summary']['QD'])
    color_m.extend([(bat_dict[i]['cycle_life'][0][0]/cyc_max)] * num)
    
x_group = np.concatenate(x_group)
y_group = np.concatenate(y_group)
color_m = np.array(color_m)


#绘制图像
fig = plt.figure(figsize=(10,5))
ax= plt.axes()
plt.scatter(x_group, y_group, s=12, c=color_m, cmap='Spectral_r') # RdBu_r
ax.set_xlim(0, 1000)
ax.set_ylim(0.95, 1.1)    
plt.xlabel('Cycle Number')
plt.ylabel('Discharge Capacity (Ah)')


sm = plt.cm.ScalarMappable(cmap='Spectral_r', norm=plt.Normalize(vmin=0, vmax=cyc_max))
cb = plt.colorbar(sm)
cb.set_label('Cycle Life')
plt.show()


# 仅展示前100圈
fig = plt.figure(figsize=(5,3))
ax= plt.axes()
plt.scatter(x_group, y_group, s=10, c=color_m, cmap='Spectral_r')
ax.set_xlim(0, 100)
ax.set_ylim(1.0, 1.1)    
plt.xlabel('Cycle Number')
plt.ylabel('Discharge Capacity (Ah)')


sm = plt.cm.ScalarMappable(cmap='Spectral_r', norm=plt.Normalize(vmin=0, vmax=cyc_max))
cb = plt.colorbar(sm)
cb.set_label('Cycle Life')
plt.show()



# 绘制100圈容量:2圈容量
dQ_100_2 = []
Q100 = []
Q2 = []

for i in bat_dict.keys():
    dQ_100_2.append(bat_dict[i]['summary']['QD'][99] / bat_dict[i]['summary']['QD'][1])
    Q100.append(bat_dict[i]['summary']['QD'][99])
    Q2.append(bat_dict[i]['summary']['QD'][1])
    
fig = plt.figure(figsize=(4,3))
plt.hist(dQ_100_2, bins=15, color='grey', edgecolor='black')
plt.plot([1.0]*11, np.arange(0,110,10),'--',color='black')

plt.xlim(0.95, 1.02)
plt.ylim(0, 100)    
plt.ylabel('Count')
plt.xlabel('Capacity ratio, cycles 100:2')
plt.show()



# 绘制温度曲线和Tdlin
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets


def plot_T(cell=1, cycle=2):
    f, ax = plt.subplots(nrows=2, figsize=(4, 8))
    ax[0].plot(
        batch1[f"b1c{cell}"]["cycles"][f"{cycle}"]["t"],
        batch1[f"b1c{cell}"]["cycles"][f"{cycle}"]["T"]);
    ax[0].grid()
    ax[0].set_ylim([29,36])
    print("T min: ", np.min(batch1[f"b1c{cell}"]["cycles"][f"{cycle}"]["T"]))
    print("T max: ", np.max(batch1[f"b1c{cell}"]["cycles"][f"{cycle}"]["T"]))

    
    ax[1].plot(
        batch1[f"b1c{cell}"]["cycles"][f"{cycle}"]["Tdlin"]);
    ax[1].grid()
    ax[1].set_ylim([29,36])
    print("Tdlin min: ", np.min(batch1[f"b1c{cell}"]["cycles"][f"{cycle}"]["Tdlin"]))
    print("Tdlin max: ", np.max(batch1[f"b1c{cell}"]["cycles"][f"{cycle}"]["Tdlin"]))
    
interact(
    plot_T,
    cell=widgets.IntSlider(value=0, description='cell', max=40, min=0),
    cycle=widgets.IntSlider(value=1, description='cycle', max=100, min=1));

# Is Qdlin a smoothed out version of Qc?
plt.plot(batch1["b1c0"]["cycles"]["1054"]["Qc"], label="Qc")
plt.plot(batch1["b1c0"]["cycles"]["1054"]["Qdlin"], label="Qdlin")
plt.grid()
plt.legend()
plt.show()


import pandas as pd
import seaborn as sns

cycle_df = {k: batch1["b1c0"]["cycles"]["1054"][k] for k in ('I', 'Qc', 'Qd', 'T', 'V', 't')}
cycle_df = pd.DataFrame.from_dict(cycle_df)
sns.heatmap(cycle_df.corr(), annot=True)
plt.show()

lin_df = {k: batch1["b1c0"]["cycles"]["1054"][k] for k in ('Qdlin', 'Tdlin', 'dQdV')}
lin_df = pd.DataFrame.from_dict(lin_df)
sns.heatmap(lin_df.corr(), annot=True)
plt.show()



#######初步分析特征与寿命的相关性
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 其他特征
features_df = pd.DataFrame()
features_df['cell_key'] = np.array(list(bat_dict.keys()))
features_df.head()

slope_lin_fit_95_100 = np.zeros(len(bat_dict.keys()))
intercept_lin_fit_95_100 = np.zeros(len(bat_dict.keys()))

for i, cell in enumerate(bat_dict.values()):
    # Compute linear fit for cycles 2 to 100:
    q = cell['summary']['QD'][94:100].reshape(-1, 1)*1000  # discharge cappacities; q.shape = (99, 1); mAh
    X = cycle_numbers = cell['summary']['cycle'][94:100].reshape(-1, 1)  # Cylce index from 2 to 100; X.shape = (99, 1)
    
    linear_regressor_95_100 = LinearRegression()
    linear_regressor_95_100.fit(X, q)
    
    slope_lin_fit_95_100[i] = linear_regressor_95_100.coef_[0]
    intercept_lin_fit_95_100[i] = linear_regressor_95_100.intercept_
        
features_df["slope_lin_fit_95_100"] = slope_lin_fit_95_100
features_df["intercept_lin_fit_95_100"] = intercept_lin_fit_95_100
features_df.head()

###########################

scaler = StandardScaler()
# scaler = MinMaxScaler()
y_life = np.log10(np.array(cycle_life).reshape(-1,1))
y_life_original = np.array(cycle_life).reshape(-1,1)
x_Q100 = np.array(Q100).reshape(-1,1)
x_Q2 = np.array(Q2).reshape(-1,1)
x_slope_95_100 = np.array(features_df['slope_lin_fit_95_100'].values).reshape(-1,1)

#标准化
y_life_scaler = scaler.fit_transform(y_life)
x_Q100_scaler = scaler.fit_transform(x_Q100)
x_Q2_scaler = scaler.fit_transform(x_Q2)
x_s_95_100_scaler = scaler.fit_transform(x_slope_95_100)

### 线性模型
regr1 = LinearRegression()
regr2 = LinearRegression()
regr3 = LinearRegression()

regr1.fit(x_Q100_scaler, y_life_scaler)
regr2.fit(x_Q2_scaler, y_life_scaler)
regr3.fit(x_s_95_100_scaler, y_life_scaler)

y1_pred = regr1.predict(x_Q100_scaler)
y2_pred = regr2.predict(x_Q2_scaler)
y3_pred = regr3.predict(x_s_95_100_scaler)

# The coefficients
r2_1 = regr1.coef_[0][0]
r2_2 = regr2.coef_[0][0]
r2_3 = regr3.coef_[0][0]
print("Coefficients: \n", r2_1)
print("Coefficients: \n", r2_2)
print("Coefficients: \n", r2_3)

# Plot outputs
color_g = y_life_original/cyc_max

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,3),sharey=True)
for i in range(0,3):
    ax[i].set_yscale('log')
    ax[0].set_ylabel('Cycle Life')
    ax[i].set_ylim(10**2, 3.5*10**3)    
    
    if i == 0:
        ax[i].scatter(x_Q2, y_life_original, s=16, c=color_g, cmap='RdBu_r') #'RdBu_r'
        ax[i].set_xlabel('Discharge Capacity @ 2 (Ah)')
        ax[i].text(0.98,10**3, 'R2 = '+ '{:.2f}'.format(r2_2), fontsize=12)
        ax[i].set_xlim(0.95, 1.1)
        
    elif i == 1:
        ax[i].scatter(x_Q100, y_life_original, s=16, c=color_g, cmap='RdBu_r')
        ax[i].set_xlabel('Discharge Capacity @ 100 (Ah)')
        ax[i].text(0.98,10**3, 'R2 = '+ '{:.2f}'.format(r2_1), fontsize=12)
        ax[i].set_xlim(0.95, 1.1)
    else:
        ax[i].scatter(x_slope_95_100, y_life_original, s=16, c=color_g, cmap='RdBu_r')
        ax[i].set_xlabel('Slope of Discharge Capacity @ 95-100 (mAh)')
        ax[i].text(-1.5,10**3, 'R2 = '+ '{:.2f}'.format(r2_3), fontsize=12)
        ax[i].set_xlim(-2, 0.5)




#######fig.2 Q100-Q10的关系图
from scipy.stats import skew, kurtosis
#  delta_Q100-10(V)
minimum_dQ_100_10 = np.zeros(len(bat_dict.keys()))
variance_dQ_100_10 = np.zeros(len(bat_dict.keys()))
skewness_dQ_100_10 = np.zeros(len(bat_dict.keys()))
kurtosis_dQ_100_10 = np.zeros(len(bat_dict.keys()))
s_dQ_100_10 = np.zeros(len(bat_dict.keys()))

for i, cell in enumerate(bat_dict.values()):
    c10 = cell['cycles']['10']
    c100 = cell['cycles']['100']
    dQ_100_10 = c100['Qdlin'] - c10['Qdlin']
    
    # 对数特征
    minimum_dQ_100_10[i] = np.log(np.abs(np.min(dQ_100_10))) # 最小值
    variance_dQ_100_10[i] = np.log(np.var(dQ_100_10)) # 方差
    skewness_dQ_100_10[i] = np.log(np.abs(skew(dQ_100_10))) # 偏度
    kurtosis_dQ_100_10[i] = np.log(np.abs(kurtosis(dQ_100_10))) # 峰度
    # 非对数特征
    s_dQ_100_10[i] = np.var(dQ_100_10)

features_df["minimum_dQ_100_10"] = minimum_dQ_100_10
features_df["variance_dQ_100_10"] = variance_dQ_100_10
features_df["skewness_dQ_100_10"] = skewness_dQ_100_10
features_df["kurtosis_dQ_100_10"] = kurtosis_dQ_100_10
features_df["s_dQ_100_10"] = s_dQ_100_10

features_df.head()


# 线性模型
x_var_Q_100_10 = np.array(features_df["variance_dQ_100_10"].values).reshape(-1,1)

# x_var = np.log10(np.array(x_var_Q_100_10).reshape(-1,1))
x_var_Q100_10_scaler = scaler.fit_transform(x_var_Q_100_10)

regr4 = LinearRegression()

regr4.fit(x_var_Q100_10_scaler, y_life_scaler)

y4_pred = regr4.predict(x_var_Q100_10_scaler)

# The coefficients
r2_4 = regr4.coef_[0][0]

print("Coefficients: \n", r2_4)


from matplotlib import cm
colormap = cm.get_cmap(name='RdBu_r') # 获取蓝色渐近色

# 绘制图像
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,3.5))
for i in range(0,3):
    
    if i == 0:
        ax[i].plot(bat_dict['b1c6']['cycles']['10']['Qd'], 
                   bat_dict['b1c6']['cycles']['10']['V'], '-', c='black')
        
        ax[i].plot(bat_dict['b1c6']['cycles']['100']['Qd'], 
                   bat_dict['b1c6']['cycles']['100']['V'], '-', c='red')
                   
        ax[i].set_xlim(0, 1.1)
        ax[i].set_ylim(2.0, 3.5)
        ax[i].set_xlabel('Discharge Capacity (Ah)')
        ax[i].set_ylabel('Voltage (V)')
        ax[i].text(0.8,3.1, 'Cycle 10', fontsize=10)
        ax[i].text(0.5,2.9, 'Cycle 10', fontsize=10, c='red')
               
    elif i == 1:
        
        v_space = np.linspace(3.5,2,1000)   # 插值电压
        plt.figure(figsize=(8,10))
        for cell in bat_dict.values():
            c = colormap(cell['cycle_life'][0][0]/cyc_max)
            c10 = cell['cycles']['10']
            c100 = cell['cycles']['100']
            dQ_100_10 = c100['Qdlin'] - c10['Qdlin']
            ax[i].plot(dQ_100_10, v_space, c=c)
            
            ax[i].set_xlim(-0.15, 0.025)
            ax[i].set_ylim(2.0, 3.5)
            ax[i].set_xlabel('Q100 - Q10 (Ah)')
            ax[i].set_ylabel('Voltage (V)')
        
    else:
            ax2 = ax[i].scatter(np.array(features_df["s_dQ_100_10"].values).reshape(-1,1),
                          y_life_original, s=16, c=y_life_original, cmap='RdBu_r')
            
            ax[i].set_xlabel('Var(ΔQ100-10(V))')
            ax[i].set_ylabel('Cycle Life')
            ax[i].text(10**-5,0.2*10**3, 'R2 = '+ '{:.2f}'.format(r2_4), fontsize=12)
            ax[i].set_xscale('log')
            ax[i].set_yscale('log')
            ax[i].set_xlim(10**-6, 10**-2)
            ax[i].set_ylim(10**2, 3.5*10**3)
            fig.colorbar(ax2, ax=ax[i], label='Cycle Life')
                   































  







        