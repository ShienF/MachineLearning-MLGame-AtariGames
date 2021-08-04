# -*- coding: utf-8 -*-
# %%
from matplotlib import pyplot as plt
import pprint
import pickle
import os
import pprint
from alive_progress import alive_bar
from os import removedirs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR

# exhaustion
Ball_pos = []

r = round(45/30, 2)
for y in range(1, 401):
    for i in range(200):
        k = 0
        l = 0
        m = 0
        n = 0
        for j in range(int(r)):
            k += round(7/r, 2)
            Ball_pos.append((round(k+i, 2), y, round((round(k+i, 2)-i), 2), 1))

        for j in range(int(r)):
            l += round(192/r, 2)
            Ball_pos.append((round(l+i, 2), y, round((round(l+i, 2)-i), 2), 1))

        for j in range(int(r)):
            m -= round(7/r, 2)
            Ball_pos.append((round(m+i, 2), y, round((round(m+i, 2)-i), 2), 1))

        for j in range(int(r)):
            n -= round(192/r, 2)
            Ball_pos.append((round(n+i, 2), y, round((round(n+i, 2)-i), 2), 1))

print(len(Ball_pos))
# print(Ball_pos[:100])
# print("I am a divider")
# %%
Ball_pos_F = []

# with alive_bar(len(Ball_pos)) as bar2:
for idx, i in enumerate(Ball_pos):
    # bar2()
    if (i[0]) >= 0 and (i[0]) <= 199:
        Ball_pos_F.append(i)

print(len(Ball_pos_F))

# with open(os.path.join(os.path.dirname(__file__),"Ball_pos.pickle"),"wb") as f:
#     pickle.dump(Ball_pos,f)

# %% Data integration

# data = []
# with open(os.path.join(os.path.dirname(__file__),"Ball_pos.pickle"),"rb") as f:
#     data.append(pickle.load(f))

Ball_x = []
Ball_y = []
Vector_x = []
Vector_y = []
Direction = []
Pos_pred = []

for ind, i in enumerate(Ball_pos_F):
    Ball_x.append(i[0])
    Ball_y.append(i[1])
    Vector_x.append(i[2])
    Vector_y.append(i[3])

    if Vector_x[-1] > 0:
        if Vector_y[-1] > 0:
            Direction.append(0)
        else:
            Direction.append(1)
    else:
        if Vector_y[-1] > 0:
            Direction.append(2)
        else:
            Direction.append(3)

print(len(Ball_x))


# feature
X = np.array([0, 0, 0, 0, 0])
with alive_bar(len(Ball_x)) as bar:
    for i in range(len(Ball_x)):
        bar()    
        X = np.vstack((X, [Ball_x[i], Ball_y[i], Vector_x[i], Vector_y[i], Direction[i]]))
X = X[1::]


# label
for i in range(len(Ball_x)):
    if Direction[i] == 0:
        pred = Ball_x[i] + ((400-Ball_y[i])//7) * 7
    else:
        pred = Ball_x[i] + ((400-Ball_y[i])//7) * (-7)

    q = pred // 200
    if q % 2 == 0:
        pred = abs(pred - 200*q)
    else:
        pred = 200 - abs(pred - 200*q)

    Pos_pred.append(pred)

Pos_pred = np.array(Pos_pred)
Pos_pred = Pos_pred.reshape(len(Pos_pred), 1)

print(len(Pos_pred))

Y = Pos_pred

print(X)
print(Y)

print(X.shape)
print(Y.shape)


# %% Training
from sklearn.metrics import accuracy_score
from sklearn import svm

print("It's training...")

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

rmse_tra = []
rmse_tes = []

# model = SVR(kernel='rbf', gamma=0.1, epsilon=0.1)
# with alive_bar(10) as bar:
#     for k in range(0,10):
#             bar()
#             k = k+1
model = SVR(kernel='rbf', gamma=0.001, epsilon=0.01) #KNeighborsRegressor(n_neighbors=3)
model.fit(x_train, np.ravel(y_train))

train_pred = model.predict(x_train)
mse = mean_squared_error(y_train, train_pred)
rmse_training = sqrt(mse)

test_pred = model.predict(x_test)
mse = mean_squared_error(y_test, test_pred)
rmse_testing = sqrt(mse)

rmse_tra.append(rmse_training)
rmse_tes.append(rmse_testing)

print("training set rmse= %.2f"% rmse_training,
        ", testing set rmse= %.2f "% rmse_testing)


# curve1 = pd.DataFrame(rmse_tra)  # elbow curve
# curve1.plot()

# curve2 = pd.DataFrame(rmse_tes)  # elbow curve
# curve2.plot()

# %%
# Plot

# cmap = sns.cubehelix_palette(as_cmap=True)
# f, ax = plt.subplots()
# points = ax.scatter(x_test[:, 0], x_test[:,1], c=test_pred, s=50, cmap=cmap)
# f.colorbar(points)
# plt.show()


#%% save the model
path = os.path.dirname(__file__)
path = os.path.join(path,"save")
if not os.path.isdir(path):
    os.mkdir(path)    
with open(os.path.join(os.path.dirname(__file__),'save','Exhaustion_SVR.pickle'),'wb') as f:
    pickle.dump(model,f)


