# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:28:55 2024

@author: Liu Yang

@project: 线性模型以及分类模型
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('rebuild_features.csv')
df.keys()

# 定义回归模型的特征与目标

varmod_features = ["variance_dQ_100_10"]

dismod_features = ["variance_dQ_100_10", "minimum_dQ_100_10","skewness_dQ_100_10",
                   "kurtosis_dQ_100_10", "discharge_capacity_2",
                   "diff_discharge_capacity_max_2"]

# fullmod_features = ["minimum_dQ_100_10", "variance_dQ_100_10",
#                     "slope_lin_fit_2_100", "intercept_lin_fit_2_100",
#                     "discharge_capacity_2", "mean_charge_time_2_6",
#                     "minimum_IR_2_100", "diff_IR_100_2"]

fullmod_features = ["minimum_dQ_100_10", "variance_dQ_100_10",
                    "slope_lin_fit_2_100", "intercept_lin_fit_2_100",
                    "discharge_capacity_2", "mean_charge_time_2_6",
                    "diff_IR_100_2"]

targetmod = ["cycle_life"]

# 分类目标与特征
varclf_features = ["variance_dQ_5_4"]
fullclf_features = ["minimum_dQ_5_4", "variance_dQ_5_4", "discharge_capacity_2",
                    "diff_IR_100_2"]

targetclf = ["cycle_550_clf"]


# 训练集划分
numBat1 = len([i for i in list(df.cell_key) if i[1] == "1"])
numBat2 = len([i for i in list(df.cell_key) if i[1] == "2"])
numBat3 = len([i for i in list(df.cell_key) if i[1] == "3"])
numBat = sum((numBat1,numBat2,numBat3))

test_ind = np.hstack((np.arange(0,(numBat1+numBat2),2),83))
train_ind = np.arange(1,(numBat1+numBat2-1),2)
secondary_test_ind = np.arange(numBat-numBat3,numBat);

splits = [train_ind, test_ind, secondary_test_ind]

# 定义训练集划分函数
def get_split(data, features, target, split):
    X = data.iloc[split,:].loc[:,features] # iloc索引,loc可基于标签索引
    y = data.iloc[split,:].loc[:,target]
    return X, y


# def eval_model(model, data, features, target, split):
#     rmse = list()
#     mpe = list()
#     for split in splits:
#         X, y = get_split(data, features, target, split)
        
#         scaler_x = StandardScaler()
#         scaler_y = StandardScaler()
        
#         X_scaler = scaler_x.fit_transform(X)
#         y_scaler = scaler_y.fit_transform(y)
        
#         pred = model.predict(X_scaler)
        
#         # 反标准化
#         pred_original = scaler_y.inverse_transform(pred)                
        
#         rmse.append(np.sqrt(mean_squared_error(pred_original, y)))
#         mpe.append(float(np.mean(np.abs((y - pred_original.reshape(-1,1))) / y * 100)))
#     return rmse, mpe

def eval_model(model, features, target, x_train, y_train):
    rmse = list()
    mpe = list()
    errors = list()
    
    X = features
    y = target
    y = np.log10(y)
    
    
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    scaler_x.fit(x_train)
    scaler_y.fit(np.log10(y_train))
    
    X_scaler = scaler_x.transform(X)
    # y_scaler = scaler_y.transform(y)
    
    pred = model.predict(X_scaler)
    
    # 反标准化
    pred_original = 10**scaler_y.inverse_transform(pred) # 注意target采用的是log10的对数               
        
    rmse.append(np.sqrt(mean_squared_error(pred_original, target)))
    mpe.append(float(np.mean(np.abs((target - pred_original.reshape(-1,1))) / target * 100)))
    
    errors.append((target - pred_original.reshape(-1,1))) # 绝对误差[cycles]
    
    return rmse, mpe, pred_original, errors

#################################Variance Model#################################
# Train Elastic net
x_train, y_train = get_split(df, varmod_features, targetmod, train_ind)
x_test, y_test = get_split(df, varmod_features, targetmod, test_ind)
x_valid, y_valid = get_split(df, varmod_features, targetmod, secondary_test_ind)

y_train_log = np.log10(y_train)
y_test_log = np.log10(y_test)
y_valid_log = np.log10(y_valid)
# 标准化
scaler_var_x = StandardScaler()
scaler_var_y = StandardScaler()

x_train_scaler = scaler_var_x.fit_transform(x_train)
# x_test_scaler = scaler_var_x.transform(x_test)
# x_valid_scaler = scaler_var_x.transform(x_valid)

y_train_scaler = scaler_var_y.fit_transform(y_train_log)
# y_test_scaler = scaler_var_y.transform(y_test_log)
# y_valid_scaler = scaler_var_y.transform(y_valid_log)


## 模型训练
alphas = np.linspace(0.001,0.01,20)
parameters = {
    "alpha": alphas,
    "l1_ratio": [0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75,0.8, 1.]}

enet = ElasticNet(random_state=54)
regr_net = GridSearchCV(enet, parameters, cv=4) #4折交叉验证

regr_net.fit(x_train_scaler, y_train_scaler)
# 输出最优模型
best_model_var = regr_net.best_estimator_

print('最优参数: ',regr_net.best_params_)
# print('最佳性能: ', regr_net.best_score_)
print("Elastic Net: %s" % regr_net.fit(x_train_scaler, y_train_scaler).score(x_train_scaler, y_train_scaler))

varmod_rmse, varmod_mpe, pred_train_var, var_errors_train = eval_model(best_model_var, x_train, y_train, x_train, y_train)
print("训练集RMSE：", varmod_rmse) # cycles
print("训练集MPE：", varmod_mpe) # %

varmod_rmse1, varmod_mpe1, pred_test_var, var_errors_test = eval_model(best_model_var, x_test, y_test, x_train, y_train)
print("测试集RMSE：", varmod_rmse1)
print("测试集MPE：", varmod_mpe1)

varmod_rmse2, varmod_mpe2, pred_vaild_var, var_errors_valid = eval_model(best_model_var, x_valid, y_valid, x_train, y_train)
print("验证集RMSE：", varmod_rmse2)
print("验证集MPE：", varmod_mpe2)

var_errors_sum = np.concatenate((var_errors_train[0].values, var_errors_test[0].values, var_errors_valid[0].values), axis =0)




#################################Dischage Model#################################
x_train_dis, y_train_dis = get_split(df, dismod_features, targetmod, train_ind)
x_test_dis, y_test_dis = get_split(df, dismod_features, targetmod, test_ind)
x_valid_dis, y_valid_dis = get_split(df, dismod_features, targetmod, secondary_test_ind)

y_train_dis_log = np.log10(y_train_dis)
y_test_dis_log = np.log10(y_test_dis)
y_valid_dis_log = np.log10(y_valid_dis)
# 标准化
scaler_dis_x = StandardScaler()
scaler_dis_y = StandardScaler()

x_train_dis_scaler = scaler_dis_x.fit_transform(x_train_dis)

y_train_dis_scaler = scaler_dis_y.fit_transform(y_train_dis_log)



## 模型训练
alphas = np.linspace(0.005,0.2,40)
parameters = {
    "alpha": alphas,
    "l1_ratio": [0.001, 0.01, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75,0.8, 1.]}

enet2 = ElasticNet(random_state=54, max_iter=3000)
regr_net_dis = GridSearchCV(enet2, parameters, cv=4) #4折交叉验证

regr_net_dis.fit(x_train_dis_scaler, y_train_dis_scaler)
# 输出最优模型
best_model_dis = regr_net_dis.best_estimator_

print('最优参数: ',regr_net_dis.best_params_)
# print('最佳性能: ', regr_net.best_score_)
print("Elastic Net: %s" % regr_net_dis.fit(x_train_dis_scaler, y_train_dis_scaler).score(x_train_dis_scaler, y_train_dis_scaler))


dismod_rmse, dismod_mpe, pred_train_dis, dis_errors_train = eval_model(best_model_dis, x_train_dis, y_train_dis, 
                                                                       x_train_dis, y_train_dis) # 后面的两个参数是为了统一用训练集的标准化规则，标准化测试集和验证集
print("训练集RMSE：", dismod_rmse) # cycles
print("训练集MPE：", dismod_mpe) # %

dismod_rmse1, dismod_mpe1, pred_test_dis, dis_errors_test = eval_model(best_model_dis, x_test_dis, y_test_dis, x_train_dis, y_train_dis)
print("测试集RMSE：", dismod_rmse1)
print("测试集MPE：", dismod_mpe1)

dismod_rmse2, dismod_mpe2, pred_vaild_dis, dis_errors_valid = eval_model(best_model_dis, x_valid_dis, y_valid_dis, x_train_dis, y_train_dis)
print("验证集RMSE：", dismod_rmse2)
print("验证集MPE：", dismod_mpe2)

dis_errors_sum = np.concatenate((dis_errors_train[0].values, dis_errors_test[0].values, dis_errors_valid[0].values), axis =0)




#################################Full Model#################################
x_train_full, y_train_full = get_split(df, fullmod_features, targetmod, train_ind)
x_test_full, y_test_full = get_split(df, fullmod_features, targetmod, test_ind)
x_valid_full, y_valid_full = get_split(df, fullmod_features, targetmod, secondary_test_ind)

y_train_full_log = np.log10(y_train_full)
y_test_full_log = np.log10(y_test_full)
y_valid_full_log = np.log10(y_valid_full)
# 标准化
scaler_full_x = StandardScaler()
scaler_full_y = StandardScaler()

x_train_full_scaler = scaler_full_x.fit_transform(x_train_full)

y_train_full_scaler = scaler_full_y.fit_transform(y_train_full_log)

## 模型训练
alphas = np.linspace(0.001,0.01,20)
parameters = {
    "alpha": alphas,
    "l1_ratio": [0.001, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75,0.8, 1.]}

enet3 = ElasticNet(random_state=54, max_iter=3000)
regr_net_full = GridSearchCV(enet3, parameters, cv=4) #4折交叉验证

regr_net_full.fit(x_train_full_scaler, y_train_full_scaler)
# 输出最优模型
best_model_full = regr_net_full.best_estimator_

print('最优参数: ',regr_net_full.best_params_)
# print('最佳性能: ', regr_net.best_score_)
print("Elastic Net: %s" % regr_net_full.fit(x_train_full_scaler, y_train_full_scaler).score(x_train_full_scaler, y_train_full_scaler))


fullmod_rmse, fullmod_mpe, pred_train_full, full_errors_train = eval_model(best_model_full, x_train_full, y_train_full, 
                                                                       x_train_full, y_train_full) # 后面的两个参数是为了统一用训练集的标准化规则，标准化测试集和验证集
print("训练集RMSE：", fullmod_rmse) # cycles
print("训练集MPE：", fullmod_mpe) # %

fullmod_rmse1, fullmod_mpe1, pred_test_full, full_errors_test = eval_model(best_model_full, x_test_full, y_test_full, x_train_full, y_train_full)
print("测试集RMSE：", fullmod_rmse1)
print("测试集MPE：", fullmod_mpe1)

fullmod_rmse2, fullmod_mpe2, pred_vaild_full, full_errors_valid = eval_model(best_model_full, x_valid_full, y_valid_full, x_train_full, y_train_full)
print("验证集RMSE：", fullmod_rmse2)
print("验证集MPE：", fullmod_mpe2)

full_errors_sum = np.concatenate((full_errors_train[0].values, full_errors_test[0].values, full_errors_valid[0].values), axis =0)


# 可视化
import matplotlib as mpl
import matplotlib.pyplot as plt

c_train = [[152/265,154/265,156/265]] * len(y_train)
c_test = [[247/265,208/265,141/265]] * len(y_test)
c_valid = [[191/265,131/265,165/265]] * len(y_valid)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,3),sharey=True)
for i in range(0,3):
    
    ax[0].set_ylabel('Predicted Cycle Life')
    ax[i].set_xlabel('Real Cycle Life')
    ax[i].set_ylim(-100, 2500)
    ax[i].set_xlim(-100, 2500)
    
    xx = np.linspace(-100, 2500, 15)
    yy = xx
    
    if i == 0:
        ax[i].scatter(y_train, pred_train_var, s=20, c=c_train, marker='o')
        ax[i].scatter(y_test, pred_test_var, s=20, c=c_test, marker='s')
        ax[i].scatter(y_valid, pred_vaild_var, s=20, c=c_valid, marker='^')
        
        ax[i].plot(xx, yy, '-', c='black')
        
        ax_hist = ax[i].inset_axes((0.6, 0.1, 0.38, 0.25))
        ax_hist.hist(var_errors_sum, bins=20, color='grey', edgecolor='black')
        ax_hist.set_xlim([-900,900])
        ax_hist.set_ylim([0,25])
        

    if i == 1:
        ax[i].scatter(y_train_dis, pred_train_dis, s=20, c=c_train, marker='o')
        ax[i].scatter(y_test_dis, pred_test_dis, s=20, c=c_test, marker='s')
        ax[i].scatter(y_valid_dis, pred_vaild_dis, s=20, c=c_valid, marker='^')
        
        ax[i].plot(xx, yy, '-', c='black')
        
        ax_hist = ax[i].inset_axes((0.6, 0.1, 0.38, 0.25))
        ax_hist.hist(dis_errors_sum, bins=30, color='grey', edgecolor='black')
        ax_hist.set_xlim([-900,900])
        ax_hist.set_ylim([0,40])
        
    if i == 2:
        ax[i].scatter(y_train_full, pred_train_full, s=20, c=c_train, marker='o')
        ax[i].scatter(y_test_full, pred_test_full, s=20, c=c_test, marker='s')
        ax[i].scatter(y_valid_full, pred_vaild_full, s=20, c=c_valid, marker='^')
        
        ax[i].plot(xx, yy, '-', c='black')
        
        ax_hist = ax[i].inset_axes((0.6, 0.1, 0.38, 0.25))
        ax_hist.hist(full_errors_sum, bins=25, color='grey', edgecolor='black')
        ax_hist.set_xlim([-1000,1000])
        ax_hist.set_ylim([0,30])
        
    plt.legend(['Fit','Train','Test','Valid'], edgecolor='none')
        

# 保存回归结果
EN_df = pd.DataFrame({"Model":["Variance model", "Discharge model", "Full model"],
              "RMSE - Train": [varmod_rmse, dismod_rmse, fullmod_rmse],
              "RMSE - Primary test": [varmod_rmse1, dismod_rmse1, fullmod_rmse1],
              "RMSE - Secondary test": [varmod_rmse2, dismod_rmse2, fullmod_rmse2],
              "MPE - Train": [varmod_mpe, dismod_mpe, fullmod_mpe],
              "MPE - Primary test": [varmod_mpe1, dismod_mpe1, fullmod_mpe1],
              "MPE - Secondary test": [varmod_mpe2, dismod_mpe2, fullmod_mpe2]})     

EN_df.to_csv('ElasticNet_reults.csv', index=False)






##############################分类模型，预测寿命长短###############################

def eval_classifier(model, data, features, target, splits):
    acc = list()    
    for split in splits:
        X, y = get_split(data, features, target, split)
        pred = model.predict(X)
        acc.append(accuracy_score(pred, y.values.ravel()))
    return acc

##逻辑分类
# variance model
x_train, y_train = get_split(df, varclf_features, targetclf, train_ind)
x_test, y_test = get_split(df, varclf_features, targetclf, test_ind)
x_valid, y_valid = get_split(df, varclf_features, targetclf, secondary_test_ind)

parameters = {"C": [0.01,0.1,0.5,0.75,1]}

logreg = LogisticRegression(solver="liblinear", random_state=54)
clf_var = GridSearchCV(logreg, parameters, cv=4)
clf_var.fit(x_train,y_train.values.ravel())

best_classifier_var = clf_var.best_estimator_
print('最优参数: ', clf_var.best_params_)
print("Logreg: %s" % clf_var.fit(x_train, y_train.values.ravel()).score(x_train, y_train.values.ravel()))

varclf_acc = eval_classifier(best_classifier_var, df, varclf_features, targetclf, splits)
print("训练集准确率：", varclf_acc) 

y_pred_var = best_classifier_var.predict(x_train), best_classifier_var.predict(x_test), best_classifier_var.predict(x_valid)



##############################################################################################################
# full model
x_train_full, y_train_full = get_split(df, fullclf_features, targetclf, train_ind)
x_test_full, y_test_full = get_split(df, fullclf_features, targetclf, test_ind)
x_valid_full, y_valid_full = get_split(df, fullclf_features, targetclf, secondary_test_ind)

parameters = {"C": [0.01,0.02,0.1,0.15,0.2,0.5,0.75,1]}

logreg1 = LogisticRegression(solver="liblinear", random_state=54)
clf_full = GridSearchCV(logreg1, parameters, cv=4)
clf_full.fit(x_train_full, y_train_full.values.ravel())

best_classifier_full = clf_full.best_estimator_
print('最优参数: ', clf_full.best_params_)
print("Logreg: %s" % clf_full.fit(x_train_full, y_train_full.values.ravel()).score(x_train_full, y_train_full.values.ravel()))

fullclf_acc = eval_classifier(best_classifier_full, df, fullclf_features, targetclf, splits)
print("训练集准确率：", fullclf_acc) 

y_pred_full = best_classifier_full.predict(x_train_full), best_classifier_full.predict(x_test_full), best_classifier_full.predict(x_valid_full)

cf_df = pd.DataFrame({"Classifier":["Variance classifier", "Full classifier"],
              "Acc - Train": [varclf_acc[0],fullclf_acc[0]],
              "Acc - Primary test": [varclf_acc[1],fullclf_acc[1]],
              "Acc - Secondary test": [varclf_acc[2],fullclf_acc[2]]})           

cf_df.to_csv('classifier_reults.csv', index=False)



























