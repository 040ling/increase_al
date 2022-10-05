import Geometry2D as geo2D
import TargetScore as ts
import sys, json, time
import base64


# 打印使用这个函数，否则前端乱码
"""
def PRINT(content):
    sys.stdout.write(base64.b64encode((content+'\n').encode('utf-8')).decode('utf-8'))
    sys.stdout.flush()
"""

def PRINT(content):
    print(content)

def compare_hit(hit1,hit2):
    len1 = hit1.len
    len2 = hit2.len
    if len1>len2:
        hit1.increase_rep()
        return hit1,hit2
    elif len1<len2:
        hit2.increase_rep()
        return hit2,hit1
    else:
        return hit1,hit2


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

    def hit_print(self):
        for hit in self.hits_list:
            hit_ = [hit.point,hit.score]
            print(hit_)

    def add_hit(self,hit,tid,status,client):
        str = "本次命中环数：{}环".format(hit.score)
        PRINT(str)
        if status == 0:
            client.publish("targetscore", json.dumps({"taskId": tid, "status": 0}))
            status += 1
        client.publish("targetscore", json.dumps({"taskId": tid, "status": 1, "score": hit.score}))
        self.hits_list.append(hit)
        return status

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




