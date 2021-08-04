#%%
import pickle
import os
import numpy as np
import random
from sklearn.neighbors import KNeighborsRegressor


class MLPlay:
    def __init__(self):
       
        self.ball_served = False
        self.previous_ball = (0,0)
        with open(os.path.join(os.path.dirname(__file__),'save','KNN_R_2300.pickle'),'rb') as f:
            self.model = pickle.load(f)
        
    def update(self, scene_info):
        
        if (scene_info["status"] == "GAME_OVER" or
            scene_info["status"] == "GAME_PASS"):
            return "RESET"
        if not self.ball_served:
           self.ball_served = True
           command = "SERVE_TO_RIGHT"
        else:
            Ball_x = scene_info["ball"][0]
            Ball_y = scene_info["ball"][1]
            Vector_x = scene_info["ball"][0] - self.previous_ball[0]
            Vector_y = scene_info["ball"][1] - self.previous_ball[1]
            Platform = scene_info["platform"][0]
            if Vector_x > 0:
                if Vector_y > 0: Direction = 0

                else: Direction = 1
            else:
                if Vector_y > 0: Direction = 2
                else: Direction = 3
            
            X = np.array([Ball_x, Ball_y, Vector_x, Vector_y, Direction]).reshape((1, -1)) # 展開成一列
            y = self.model.predict(X)

            # print(y)

            if scene_info["platform"][0]+20 + 5 < y:
                command = "MOVE_RIGHT"
            elif scene_info["platform"][0]+20 - 5  > y:
                command = "MOVE_LEFT"
            else:
                command = random.choice(("MOVE_RIGHT","MOVE_LEFT","NONE"))

        self.previous_ball = scene_info["ball"]
        return command

    def reset(self):
        self.ball_served = False    


# %%
