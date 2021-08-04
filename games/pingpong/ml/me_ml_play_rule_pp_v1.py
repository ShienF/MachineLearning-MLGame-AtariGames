import random

class MLPlay:
    def __init__(self, side):
        """
        Constructor

        @param side A string "1P" or "2P" indicates that the `MLPlay` is used by
               which side.
        """
        self.ball_served = False
        self.side = side
        self.previous_ball = (0,0)
        self.pred = 100 # 預設板子x位置為80
        

    def update(self, scene_info):
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"

        if not self.ball_served:
            self.ball_served = True
            self.previous_ball = scene_info["ball"]
            command = "SERVE_TO_RIGHT" ##
            
        else: #rule code
            self.speed = scene_info["ball_speed"][1]
            i = self.speed
            i = abs(i)
            self.pred = 100 #回到中心點
            
            if self.side == "1P":
                if self.previous_ball[1] - scene_info["ball"][1] > 0:
                    if scene_info["ball"][1] >= 260:
                        self.pred = (scene_info["ball"][0] + ((scene_info["ball"][1] - 260) // i) * (scene_info["ball"][0] - self.previous_ball[0])) + ((160//i)*(scene_info["ball"][0] - self.previous_ball[0])) #160 = 420-260
                    else:
                        pass
                else:
                    self.pred = scene_info["ball"][0] + ((420 - scene_info["ball"][1]) // i) * (scene_info["ball"][0] - self.previous_ball[0])
                
                    
                 # 運用反射原理調整板子預測位置
                q = self.pred // 200
                if (q % 2 == 0):
                    self.pred = abs(self.pred - 200*q)
                else:
                    self.pred = 200 - abs(self.pred - 200*q)
                
                # 板寬40, +20為中心點
                if scene_info["platform_1P"][0] + 20 +5 < self.pred:
                    command = "MOVE_RIGHT"
                elif scene_info["platform_1P"][0] + 20 -5 > self.pred:
                    command = "MOVE_LEFT"
                else:
                    command = random.choice(("NONE","MOVE_LEFT","MOVE_RIGHT"))      
                    
            else:
                if self.previous_ball[1] - scene_info["ball"][1] < 0:
                    if scene_info["ball"][1] <= 240:
                        self.pred = (scene_info["ball"][0] + ((240 - scene_info["ball"][1]) // i) * (scene_info["ball"][0] - self.previous_ball[0])) + ((160//i )*(scene_info["ball"][0] - self.previous_ball[0]))
                    else:
                        pass
                else:
                    self.pred = scene_info["ball"][0] + ( (scene_info["ball"][1] - 80) // i ) * (scene_info["ball"][0] - self.previous_ball[0])
                
                    
                # 運用反射原理調整板子預測位置
                q = self.pred // 200
                if (q % 2 == 0):
                    self.pred = abs(self.pred - 200*q)
                else:
                    self.pred = 200 - abs(self.pred - 200*q)
                
                # 板寬40, +20為中心點
                if scene_info["platform_2P"][0] + 20 +5  < self.pred:
                    command = "MOVE_RIGHT"
                elif scene_info["platform_2P"][0] + 20 -5  > self.pred:
                    command = "MOVE_LEFT"
                else:
                    command = random.choice(("NONE","MOVE_LEFT","MOVE_RIGHT")) 
                
  
            self.previous_ball = scene_info["ball"]
            
        return command

    def reset(self):
        self.ball_served = False
