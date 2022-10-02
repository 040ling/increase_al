import cv2
import numpy as np
import ImgProcess as pro
import TargetScore as ts
import queue
import threading
import time
import sys
from tool import PRINT
import VideoAnalyze as va


# 测试是从摄像头获取数据还是从历史视频获取数据
class CamCapture:
    def __init__(self, link, num):
        self.cap = cv2.VideoCapture(link)  # 通过link建立摄像头连接
        self.q = queue.Queue()  # 读入图片的队列
        self.ret = True
        self.num = num  # 正面：0 左侧：1 右侧：2
        self.link = link
        t = threading.Thread(target=self._reader, name="grandson" + str(num))  # 构造读图片的进程
        t.daemon = True
        t.start()
    #待分析的图片
    def _reader(self):
        global ExitFlag  # 指示是否已退出判别流程，如果是则不执行循环
        # 这里需要有一个开始判读 停止判读的按钮
        ExitFlag = 0  # 默认没有退出
        while True:
            try:
                self.ret, self.frame = self.cap.read()  # 拿到图片交给self.frame
            except:
                print("无法正常获取图片")
                r_channel = np.zeros((1920, 1080), dtype=np.uint8)
                g_channel = np.zeros((1920, 1080), dtype=np.uint8)
                b_channel = np.zeros((1920, 1080), dtype=np.uint8)
                self.frame = cv2.merge((b_channel, g_channel, r_channel))  # 显示为黑屏
                self.ret = False

            if not self.ret:  # 如果超过5s没拿到图片就播放黑屏
                r_channel = np.zeros((1920, 1080), dtype=np.uint8)
                g_channel = np.zeros((1920, 1080), dtype=np.uint8)
                b_channel = np.zeros((1920, 1080), dtype=np.uint8)
                self.frame = cv2.merge((b_channel, g_channel, r_channel))
                # break00
            if not self.q.empty():  # 如果当前有图片则取出图片
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(self.frame)  # 向队列中载入图片
            if ExitFlag:  # 如果触发flag则退出
                break
        return

    def read(self):
        self.get_start = time.time()
        while time.time() - self.get_start < 5:
            try:
                return True, self.q.get_nowait()
            except:
                time.sleep(0.01)

        print("超过5s没有拿到图片")
        return False, self.frame


class cam:
    def __init__(self, num, DisplayMode, path, save_path, model,bullseye, innerdist, num_target):
        self.num = num  # 摄像头号码
        self.DisplayMode = DisplayMode  # 1：文件夹视频
        if self.DisplayMode:  # 本地视频测试模式初始化
            self.fn = path
            self.cam = cv2.VideoCapture(self.fn)
            if not self.cam.isOpened():
                print("could not open video_src " + str(self.fn) + " !\n")
                sys.exit(1)
        else:  # 在线视频测试模式初始化 先拍正面的
            self.fn = "rtsp://admin:Abcd12345678@192.168.1.62" + ":554/h265/ch33/main/av_stream?tcp"
            save_vid = './' + save_path + '/' + str(120 + self.num) + '.avi'
            self.cam = CamCapture(self.fn, self.num)
            print(save_vid)
            self.vidWriter = cv2.VideoWriter(save_vid, cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 12.0,
                                             (1920, 1080))  # 服务器在线视频保存路径
        #print(self.fn)  # 读取文件的路径
        ret, prev = self.cam.read()  # 从摄像头读来的帧
        self.model = model

        self.cam_work = 1

        # 根据监控摄像头判断当前的正面 侧面等

        self.frame_a = prev
        self.frame_b = prev
        self.frame_c = prev

        self.cnt = 0
        self.video_hit =va.VideoHits()
        self.bullseye = bullseye
        self.innerdist = innerdist
        self.num_target = num_target
        self.point_a = pro.img_pts(model,self.frame_a,self.bullseye)
        self.point_b = pro.img_pts(model, self.frame_b, self.bullseye)

    def cam_detect(self):
        ret, frame = self.cam.read()

        if not ret:
            self.cam_work = 0
            print('读取摄像头{}/文件夹视频失败'.format(self.num))

        """"""

        self.frame_c = frame
        #self.frame_a,self.frame_b=shipin.chulishipin(self.model,self.frame_a,self.frame_b,self.frame_c)
        self.frame_a,self.frame_b,self.video_hit,self.point_a,self.point_b=ts.video_process(self.model,self.frame_a,self.frame_b,self.frame_c,self.video_hit,
                                                                  self.bullseye,self.innerdist,self.num_target,self.point_a,self.point_b)

            #self.video_hit = va.VideoHits  # 重新初始化


    def video_detect(self,i,t1):
        ret, frame = self.cam.read()

        if not ret:
            self.cam_work = 0
            print('读取摄像头{}/文件夹视频失败'.format(self.num))
            return 0,t1

        if i != 15:
            return 1,t1

        arch_num = self.video_hit.get_num()

        self.frame_c = frame
        #self.frame_a,self.frame_b=shipin.chulishipin(self.model,self.frame_a,self.frame_b,self.frame_c)
        self.frame_a,self.frame_b,self.video_hit,self.point_a,self.point_b=ts.video_process(self.model,self.frame_a,self.frame_b,self.frame_c,self.video_hit,
                                                                  self.bullseye,self.innerdist,self.num_target,self.point_a,self.point_b,t1)

        arch_num_hou = self.video_hit.get_num()
        if arch_num_hou-arch_num!=0:
            t1 = time.time()



            #self.video_hit=va.VideoHits()  # 重新初始化

        return 1,t1






