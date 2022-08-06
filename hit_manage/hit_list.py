import sys
import time
sys.path.append("..")

import Geometry2D as geo2D
from tool import PRINT




class VideoHits:
    def __init__(self):
        self.hits_list=[]
    def check_hit(self,new_hit):
        dmin=100000
        for hit in self.hits_list:
            point0=hit.point
            point1=new_hit.point
            d = geo2D.euclidean_dist(point0,point1)  # 算和每点的距离
            if d<dmin:
                dmin = d
        return dmin

    def add_hit(self,hit):
        str = "本次命中环数：{}环".format(hit.score)
        PRINT(str)
        time.sleep(0.5)
        self.hits_list.append(hit)

    def final_pre(self,score):
        """
        这个函数单纯对比分数
        Args:
            score:

        Returns:

        """
        my_score = 0
        for hit in self.hits_list:
            my_score += hit.score
        str = "总环数：{}/{}".format(my_score,len(self.hits_list)*10)
        win_loss = "你赢了" if my_score>score else "你输了"
        PRINT(str)
        time.sleep(0.5)
        PRINT(win_loss)

    def s_score(self,score):
        return

    def get_num(self):
        return len(self.hits_list)




