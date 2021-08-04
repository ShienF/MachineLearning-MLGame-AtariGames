#%%
import pickle
import numpy as np # 數據科學
import pandas as pd 
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from alive_progress import alive_bar
import pprint
from sklearn.model_selection import GridSearchCV


path = os.path.join(os.path.dirname(__file__),"..","log") # 結合此檔案及log的路徑成為log的絕對路徑
allFile = os.listdir(path) # 列出所有log內檔案列表
data_set = []
for file in allFile:
    with open(os.path.join(path,file),"rb") as f: # 以二進制方式讀取檔案
        data_set.append(pickle.load(f)) # 將f的每份pickle內資料加入至data_set串列中

Ball_x = []
Ball_y = []
Vector_x = []
Vector_y = []
Direction = []

for data in data_set:
    for i, sceneInfo in enumerate(data['ml']["scene_info"][2:-2]): # 第0筆資訊不要, -2:下述i的關係
        Ball_x.append(sceneInfo['ball'][0])
        Ball_y.append(sceneInfo['ball'][1])
        Vector_x.append(data['ml']['scene_info'][i+2]["ball"][0]-data['ml']['scene_info'][i+1]["ball"][0]) # 取得前後frame的x座標差
        Vector_y.append(data['ml']['scene_info'][i+2]["ball"][1]-data['ml']['scene_info'][i+1]["ball"][1])
        if Vector_x[-1] > 0:
            if Vector_y[-1] > 0: Direction.append(0) # 球往右下
            else: Direction.append(1) # 球往右上
        else :
            if Vector_y[-1] > 0: Direction.append(2) # 球往左下
            else: Direction.append(3) # 球往左上

# taking out identical data           
data_group = []
for i in range(len(Ball_x)):
    data_group.append((Ball_x[i],Ball_y[i],Vector_x[i],Vector_y[i],Direction[i]))
data_group_set = set(data_group)
data_group_list = list(data_group_set)
print(len(data_group),len(data_group_list))

Ball_x = []
Ball_y = []
Vector_x = []
Vector_y = []
Platform = []
Direction = []
Command = []
cut = 0
for i in range(len(data_group_list)):
    Ball_x.append(data_group_list[i][0])
    Ball_y.append(data_group_list[i][1])
    Vector_x.append(data_group_list[i][2])
    Vector_y.append(data_group_list[i][3])
    Direction.append(data_group_list[i][4])
    if abs(Vector_x[i]) != abs(Vector_y[i]):
        cut = cut+1
print(len(Ball_x),cut,'cut rate={:.2%}'.format(cut/len(Ball_x)))    

#%%     

# feature

X = np.array([0,0,0,0,0])
with alive_bar(len(Ball_x)) as bar:
    for i in range(len(Ball_x)):
        bar()    
        X = np.vstack((X, [Ball_x[i], Ball_y[i], Vector_x[i], Vector_y[i], Direction[i]]))
X = X[1::]


# label
Pos_pred = [] #預測落點位置
for i in range(len(Ball_x)):
    if Direction[i] == 0 or Direction[i] == 1:
        pred = Ball_x[i] + ((400-Ball_y[i])//7) * 7
    else:
        pred = Ball_x[i] + ((400-Ball_y[i])//7) * (-7)

    q = (pred // 200)
    if (q % 2 == 0):
        pred = abs(pred - 200*q)
    else:
        pred = 200 - abs(pred - 200*q)

    Pos_pred.append(pred)

Pos_pred = np.array(Pos_pred)
Y = Pos_pred

print(X)
print(Y)

print(X.shape)
print(Y.shape)

# #%% 標準化
# sc_x = StandardScaler()
# sc_y = StandardScaler()

# X = sc_x.fit_transform(X)
# Y = sc_y.fit_transform(Y)

#%% training

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2) # 資料拆成8:2的訓練及測試
# param_grid = {'kernel':['rbf'],'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001],'epsilon':[1,0.1,0.01,.001]}
# grid = GridSearchCV(SVR(),param_grid,refit=True,verbose=2)
# grid.fit(x_train,y_train)
# print(grid.best_estimator_)

# model = SVR(kernel='rbf', C=100, gamma=0.001, epsilon=1)
# model.fit(x_train, y_train)

# train_pred = model.predict(x_train)
# mse = mean_squared_error(y_train, train_pred)
# rmse_training = sqrt(mse)

# test_pred = model.predict(x_test)
# mse = mean_squared_error(y_test, test_pred)
# rmse_testing = sqrt(mse)

# print(" training set rmse = %.2f" % rmse_training,
#         ", testing set rmse = %.2f" % rmse_testing)




rmse_tra = []
rmse_tes = []

# # with alive_bar(10) as bar:
# for k in range(0,80):
#         # bar()
#     k = k+1
#     model = KNeighborsRegressor(n_neighbors=k) 
#     model.fit(x_train, y_train)

#     train_pred = model.predict(x_train)
#     mse = mean_squared_error(y_train, train_pred)
#     rmse_training = sqrt(mse)
#     rmse_tra.append(rmse_training)

#     test_pred = model.predict(x_test)
#     mse = mean_squared_error(y_test, test_pred)
#     rmse_testing = sqrt(mse)
#     rmse_tes.append(rmse_testing)

# plt.figure(figsize=(10,6))
# plt.plot(range(0,80), rmse_tra, color='blue', marker='o', markerfacecolor='blue', markersize=5)
# plt.plot(range(0,80), rmse_tes, color='green', marker='o', markerfacecolor='green', markersize=5)
# plt.title('RMSE v.s K value')
# plt.xlabel=('K')
# plt.ylabel=('RMSE')
# print('min RMSE= %.2f' %min(rmse_tes),'at K=',rmse_tes.index(min(rmse_tes)))
# plt.show()


# model = KNeighborsClassifier(n_neighbors=rmse_tes.index(min(rmse_tes)))
# model.fit(x_train,y_train)

# train_pred = model.predict(x_train)
# mse = mean_squared_error(y_train, train_pred)
# rmse_training = sqrt(mse)

# test_pred = model.predict(x_test)
# mse = mean_squared_error(y_test, test_pred)
# rmse_testing = sqrt(mse)

# print("k=",rmse_tes.index(min(rmse_tes))," training set rmse = %.2f" % rmse_training,
#         ", testing set rmse = %.2f" % rmse_testing)

model = KNeighborsRegressor(n_neighbors=2) #SVR(kernel='rbf',degree=3, gamma=0.001, epsilon=0.01)
model.fit(x_train, y_train)

train_pred = model.predict(x_train)
mse = mean_squared_error(y_train, train_pred)
rmse_training = sqrt(mse)

test_pred = model.predict(x_test)
mse = mean_squared_error(y_test, test_pred)
rmse_testing = sqrt(mse)

print("k=2"," training set rmse = %.2f" % rmse_training,", testing set rmse = %.2f" % rmse_testing)

# save the model
path = os.path.dirname(__file__)
path = os.path.join(path,"save")
if not os.path.isdir(path):
    os.mkdir(path)                                                                                                                                                        
    
with open(os.path.join(os.path.dirname(__file__),'save',"KNN_R_test.pickle"),'wb') as f:
    pickle.dump(model,f)



# rmse_tra.append(rmse_training)
# rmse_tes.append(rmse_testing)

# curve1 = pd.DataFrame(rmse_tra)  # elbow curve
# curve1.plot()

# curve2 = pd.DataFrame(rmse_tes)  # elbow curve
# curve2.plot()



#%% plot

# x_grid = np.arrange(min(X),max(X),0.01)
# x_grid = x_grid.reshape((len(x_grid),1))
# plt.scatter(X[], Y, color = 'red')
# plt.plot(x_grid, model.predict(x_grid), color = 'blue')



#%% training

# training
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2) # 資料拆成8:2的訓練及測試
model = KNeighborsRegressor(n_neighbors=k) # k值需調整
model.fit(x_train, y_train)

# evaluation
train_pred = model.predict(x_train)
mse = mean_squared_error(y_train, train_pred)
rmse_training = sqrt(mse)

test_pred = model.predict(x_test)
mse = mean_squared_error(y_test, test_pred)
rmse_testing = sqrt(mse)

# save the model                                                                                                                                                 
with open(os.path.join(os.path.dirname(__file__),'save',"存檔名稱.pickle"),'wb') as f:
    pickle.dump(model,f)


