#%% import

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random


#%% Data preprocessing

## loading pickle content to the list
path = os.path.join(os.path.dirname(__file__),"..","base_log")
allfile = os.listdir(path)
data_set = []
for file in allfile:
    with open(os.path.join(path,file),"rb") as f:
        data_set.append(pickle.load(f))

Ball_x = []
Ball_y = []
Ball_speed_x = []
Ball_speed_y = []
Platform = []
Platform_opp = []
Blocker = []
Direction = []
Command = []

for data in data_set:
    for i, value in enumerate(data['ml_2P']['scene_info'][1:-5]): # 2P data
        Ball_x.append(value['ball'][0])
        Ball_y.append(value['ball'][1])
        Ball_speed_x.append(value['ball_speed'][0])
        Ball_speed_y.append(value['ball_speed'][1])
        Platform.append(value['platform_2P'][0]) # 1P 1P[1]=420, 2P[1]=50+30
        Platform_opp.append(value['platform_1P'][0]) # 1P
        if ('blocker' in value) == False:
            Blocker.append(240) #240代表超出遊戲畫面的位置, 表示球不會碰到
        else:
            Blocker.append(value['blocker'][0]) #[1]=240
        if value['ball'][0]>0:
            if value['ball'][1]>0:
                Direction.append(0) #右下
            else: 
                Direction.append(1) #右上
        else:
            if value['ball'][1]>0:
                Direction.append(2) #左下
            else:
                Direction.append(3) #左上
    for command in data['ml_2P']['command'][1:-5]: # 2P data
        if command == "NONE":
            Command.append(0)
        elif command == "MOVE_LEFT":
            Command.append(-1)
        elif command == "MOVE_RIGHT":
            Command.append(1)
print("no. original data set=",len(Ball_x))

#%% advanced data

path = os.path.join(os.path.dirname(__file__),"..","advanced_log_2P")
allfile = os.listdir(path)
data_set = []
for file in allfile:
    with open(os.path.join(path,file),"rb") as f:
        data_set.append(pickle.load(f))

for data in data_set:
    for i in range(len(data['ml_2P']['scene_info'])-1,0,-1): #start,end,sep
        if data['ml_2P']['scene_info'][i]['ball'][1] > 80 and i >= ((len(data['ml_2P']['scene_info'])-1)-(370//abs(data['ml_2P']['scene_info'][i]['ball_speed'][1]))): #420-50=從1P板子掉下來的最後幾球 
            Ball_x.append(data['ml_2P']['scene_info'][i]['ball'][0])
            Ball_y.append(data['ml_2P']['scene_info'][i]['ball'][1])
            Ball_speed_x.append(data['ml_2P']['scene_info'][i]['ball_speed'][0])
            Ball_speed_y.append(data['ml_2P']['scene_info'][i]['ball_speed'][1])
            Platform.append(data['ml_2P']['scene_info'][i]['platform_2P'][0])
            Platform_opp.append(data['ml_2P']['scene_info'][i]['platform_1P'][0])
            if ('blocker' in data['ml_2P']['scene_info'][i]) == False:
                Blocker.append(240) #240代表超出遊戲畫面的位置, 表示球不會碰到
            else:
                Blocker.append(data['ml_2P']['scene_info'][i]['blocker'][0]) #[1]=240
            if data['ml_2P']['scene_info'][i]['ball_speed'][0]>0:
                if data['ml_2P']['scene_info'][i]['ball_speed'][1]>0:
                    Direction.append(0) #右下
                else: 
                    Direction.append(1) #右上
            else:
                if data['ml_2P']['scene_info'][i]['ball_speed'][1]>0:
                    Direction.append(2) #左下
                else:
                    Direction.append(3) #左上
            y_speed = data['ml_2P']['scene_info'][i]['ball_speed'][1]
            y_speed = abs(y_speed)
            x_speed = data['ml_2P']['scene_info'][i]['ball_speed'][0]
            pred = 100
            if data['ml_2P']['scene_info'][i]['ball_speed'][1] > 0:
                if data['ml_2P']['scene_info'][i]["ball"][1] <= 240:
                    pred = (data['ml_2P']['scene_info'][i]["ball"][0] + ((240 - data['ml_2P']['scene_info'][i]["ball"][1]) // y_speed) *  x_speed) + ((160//y_speed)* x_speed)
                else:
                    pass
            else:
                pred = data['ml_2P']['scene_info'][i]["ball"][0] + (((data['ml_2P']['scene_info'][i]["ball"][1] - 80) // y_speed) *  x_speed)

            # 運用反射原理調整板子預測位置
            q = pred // 200
            if (q % 2 == 0):
                pred = abs(pred - 200*q)
            else:
                pred = 200 - abs(pred - 200*q)

            if data['ml_2P']['scene_info'][i]["platform_2P"][0] + 20 +5 < pred:
                    Command.append(1)
            elif data['ml_2P']['scene_info'][i]["platform_2P"][0] + 20 -5 > pred:
                Command.append(-1)
            else:
                Command.append(random.choice((0,1,-1)))
        elif i < ((len(data['ml_2P']['scene_info'])-1)-(370//abs(data['ml_2P']['scene_info'][i]['ball_speed'][1]))): 
            break
            
            
print("no. data set plus advanced data=",len(Ball_x))

#%%
## taking out identical data
data_group = []
for i in range(len(Ball_x)):
    data_group.append((Ball_x[i],Ball_y[i],Ball_speed_x[i],Ball_speed_y[i],Platform[i],Platform_opp[i],Blocker[i],Command[i]))
data_group_set = set(data_group)
data_group_list = list(data_group_set)
print(len(data_group),len(data_group_list))


Ball_x = []
Ball_y = []
Ball_speed_x = []
Ball_speed_y = []
Platform = []
Platform_opp = []
Blocker = []
Command = []
cut = 0
for i in range(len(data_group_list)):
    Ball_x.append(data_group_list[i][0])
    Ball_y.append(data_group_list[i][1])
    Ball_speed_x.append(data_group_list[i][2])
    Ball_speed_y.append(data_group_list[i][3])
    Platform.append(data_group_list[i][4])
    Platform_opp.append(data_group_list[i][5])
    Blocker.append(data_group_list[i][6])
    Command.append(data_group_list[i][7])
    if abs(Ball_speed_x[i]) != abs(Ball_speed_y[i]):
        cut = cut+1
print(len(Ball_x),cut,'cut rate={:.2%}'.format(cut/len(Ball_x)))


#%% features and label

X = np.array([0,0,0,0,0,0])
for i in range(len(Ball_x)):
    X = np.vstack((X,[Ball_x[i],Ball_y[i],Ball_speed_x[i],Ball_speed_y[i],Platform[i],Blocker[i]])) #Platform_opp[i]
X = X[1::]
print(X.shape)

Command = np.array(Command)
# Command = Command.reshape(len(Command),1)

Y = Command
print(Y.shape)

#%% training

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
acc = []
for k in range(0,40):
    k = k+1
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train,y_train)
    y_predict = model.predict(x_test)
    # print("k=",k,"Accuracy=",accuracy_score(y_predict,y_test))
    acc.append(accuracy_score(y_predict,y_test))

plt.figure(figsize=(10,6))
plt.plot(range(0,40), acc, color='blue', marker='o', markerfacecolor='red', markersize=10)
plt.title('Accuracy v.s K value')
plt.xlabel=('K')
plt.ylabel=('Accuracy')
print('Max accuracy=',max(acc),'at K=',acc.index(max(acc)))
plt.show()

model = KNeighborsClassifier(n_neighbors=acc.index(max(acc)))
model.fit(x_train,y_train)
y_predict = model.predict(x_test)
print("k=",acc.index(max(acc)),"Accuracy=",accuracy_score(y_predict,y_test))

#%% save the model

with open(os.path.join(os.path.dirname(__file__),'save','KNN_v1_lastout_2P_adv15.pickle'),'wb') as f:
    pickle.dump(model,f)
