
class Hit:
    def __init__(self,x,y,d,score,len,bullseye):
        """

        :param x: 横坐标
        :param y: 纵坐标
        :param d: 圆心距离
        :param score: 环数
        :param len: 箭长
        :param bullseye: 圆心坐标
        """
        self.point = (x,y)
        self.score = score
        self.d = d
        self.bullseye = bullseye
        self.reputation = 1
        self.len = len

    def increase_rep(self):
        self.reputation+=1

    def descrease_rep(self):
        self.reputation-=1






