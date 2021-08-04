"""
Playing game by KNN model.
"""
import pickle
import os
import numpy as np


class MLPlay:
    def __init__(self, side):
       
        self.ball_served = False
        self.side = side
        # self.previous_ball = (0,0)
        with open(os.path.join(os.path.dirname(__file__),'save','KNN_v1_lastout_adv10.pickle'),'rb') as f:
            self.model = pickle.load(f)
        
    def update(self, scene_info):
        
        if (scene_info["status"] == "GAME_1P_WIN" or
            scene_info["status"] == "GAME_2P_WIN" or
            scene_info["status"] == "GAME_DRAW"):
            return "RESET"
        if not self.ball_served:
           self.ball_served = True
           command = "SERVE_TO_RIGHT"
        else:
            Ball_x = scene_info["ball"][0]
            Ball_y = scene_info["ball"][1]
            Ball_speed_x = scene_info["ball_speed"][0]
            Ball_speed_y = scene_info["ball_speed"][1]
            Platform = scene_info["platform_1P"][0]
            if ('blocker' in scene_info) == False:
                Blocker = 240
            else:
                Blocker = scene_info["blocker"][0]
        
            # if ['ball'][0] > 0:
            #     if ['ball'][1] > 0: Direction = 0
            #     else: Direction = 1
            # else:
            #     if ['ball'][1] > 0: Direction = 2
            #     else: Direction = 3
            
            x = np.array([Ball_x, Ball_y, Ball_speed_x, Ball_speed_y, Platform, Blocker]).reshape((1, -1)) # 展開成一列
            y = self.model.predict(x)
            if y == 0: command = "NONE"
            elif y == -1: command = "MOVE_LEFT"
            elif y == 1: command = "MOVE_RIGHT"

        # self.previous_ball = scene_info["ball"]
        return command

    def reset(self):
        self.ball_served = False    
        
