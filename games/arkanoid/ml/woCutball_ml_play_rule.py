# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 17:44:54 2021

@author: user
"""


class MLPlay:
    def __init__(self):
        self.ball_served = False
        self.previous_ball = (0,0)
        self.pred = 100 # 預設板子x位置為100
    
    def update(self, scene_info):
        if (scene_info["status"] == "GAME_OVER" or
            scene_info["status"] == "GAME_PASS"):
            return "RESET"
        
        if not self.ball_served:
            self.ball_served = True # 發球未?
            self.previous_ball = scene_info["ball"]
            command = "SERVE_TO_LEFT"
        
        else:
            # rule code
            self.pred = 100 
            if self.previous_ball[1] - scene_info["ball"][1] > 0: # 球正在往上 [0]=x座標 [1]=y座標
                pass
            else: # 球正在往下，判斷球的落點; 每frame球移動速度為+-7
                self.pred = scene_info["ball"][0] + ((400 - scene_info["ball"][1]) // 7 ) * (scene_info["ball"][0] - self.previous_ball[0])
                
            # 反射原理, 調整self.pred   
            q = self.pred // 200
            if (q % 2 == 0):
                self.pred = abs(self.pred - 200*q)
            else:
                self.pred = 200 - abs(self.pred - 200*q)
            
            
            # 板寬40, +20為中心點, +-5為板子每frame移動速度
            if scene_info["platform"][0]+20 +5 < self.pred:
                command = "MOVE_RIGHT"
            elif scene_info["platform"][0]+20 -5 > self.pred:
                command = "MOVE_LEFT"
            else:
                command = "NONE"
               
        self.previous_ball = scene_info["ball"]
        return command
    
    def reset(self):
        self.ball_served = False