import pickle
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

# I put this program in the same folder as MLGame/games/arkaonid/ml
# you can edit path to get log folder

path = os.path.join(os.path.dirname(__file__), "..", "log")

allFile = os.listdir(path)
data_set = []
for file in allFile:
    with open(os.path.join(path, file), "rb") as f:
        data_set.append(pickle.load(f))


Ball_x = []
Ball_y = []
Ball_speed_x = []
Ball_speed_y = []
Direction = []
Platform = []
Command = []

for data in data_set:
    for i, sceneInfo in enumerate(data['ml']['scene_info'][1:-2]):
        Ball_x.append(sceneInfo['ball'][0])
        Ball_y.append(sceneInfo['ball'][1])
        Platform.append(sceneInfo['platform'][0])
        Ball_speed_x.append(data['ml']['scene_info'][i+2]["ball"][0]-data['ml']['scene_info'][i+1]["ball"][0])
        Ball_speed_y.append(data['ml']['scene_info'][i+2]["ball"][1]-data['ml']['scene_info'][i+1]["ball"][1])
        if Ball_speed_x[-1] > 0 :
            if Ball_speed_y[-1] > 0:  Direction.append(0)
            else :  Direction.append(1)
        else :
            if Ball_speed_y[-1] > 0:  Direction.append(2)
            else :  Direction.append(3)
    for command in data['ml']["command"][1:-2]:
        if command == "NONE":
            Command.append(0)
        elif command == "MOVE_LEFT":
            Command.append(-1)
        elif command == "MOVE_RIGHT":
            Command.append(1)
    
# feature
X = np.array([0,0,0,0,0,0])
for i in range(len(Ball_x)):
    X = np.vstack((X, [Ball_x[i], Ball_y[i], Direction[i], Ball_speed_x[i], Ball_speed_y[i], Platform[i]]))
X = X[1::]

# label
Y = Command

#%% training 

x_train , x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
model = svm.SVC(gamma=0.001, decision_function_shape='ovo')
#model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
#print(y_predict)
#print(y_test)
print("Accuracy(正確率) = ", accuracy_score(y_predict, y_test))

##%% save the model
path = os.path.dirname(__file__)
path = os.path.join(path,"save")
if not os.path.isdir(path):
    os.mkdir(path)

with open(os.path.join(os.path.dirname(__file__),'save',"SVCmodel.pickle"), 'wb') as f:
    pickle.dump(model,f)