import cv2
import numpy as np
import ImgProcess as pro
import TargetScore as ts
import queue
import threading
import time
import sys
import shipin
import VideoAnalyze as va


# 测试是从摄像头获取数据还是从历史视频获取数据
class CamCapture:
    def __init__(self, link, num):
        self.cap = cv2.VideoCapture(link)  # 通过link建立摄像头连接
        self.q = queue.Queue()  # 读入图片的队列
        self.ret = True
        self.num = num
        self.link = link
        t = threading.Thread(target=self._reader, name="grandson" + str(num))  # 构造读图片的进程
        t.daemon = True
        t.start()
    #待分析的图片
    def _reader(self):
        global ExitFlag  # 指示是否已退出判别流程，如果是则不执行循环
        ExitFlag = 0  # 默认没有退出
        i = 0
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
            if i == 5:
                self.q.put(self.frame)  # 向队列中载入图片
                i = 0
            else:
                i += 1
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

class VideoCapture:
    def __init__(self, fn, num):
        self.cap = cv2.VideoCapture(fn)  # 读入视频
        self.q = queue.Queue()  # 读入图片的队列
        self.ret = True
        t = threading.Thread(target=self._reader, name="grandson" + str(num))  # 构造读图片的进程
        t.daemon = True
        t.start()
    #待分析的图片
    def _reader(self):
        global ExitFlag  # 指示是否已退出判别流程，如果是则不执行循环
        ExitFlag = 0  # 默认没有退出
        i = 0
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
            if i == 5:
                self.q.put(self.frame)  # 向队列中载入图片
                i = 0
            else:
                i += 1
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
    def __init__(self, num, DisplayMode, path, save_path, fn):
        self.num = num  # 摄像头号码
        self.DisplayMode = DisplayMode
        if self.DisplayMode:  # 本地视频测试模式初始化
            self.fn = path
            self.cam = VideoCapture(self.fn)
            if not self.cam.isOpened():
                print("could not open video_src " + str(self.fn) + " !\n")
                sys.exit(1)
        else:  # 在线视频测试模式初始化 先拍正面的
            self.fn = "rtsp://admin:Abcd12345678@3.1.200." + str(self.num + 191) + ":554/h265/ch33/main/av_stream?tcp"
            save_vid = './' + save_path + '/' + str(120 + self.num) + '.avi'
            self.cam = CamCapture(self.fn, self.num)
            print(save_vid)
            self.vidWriter = cv2.VideoWriter(save_vid, cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 12.0,
                                             (640, 360))  # 服务器在线视频保存路径
        print(self.fn)
        ret, prev = self.cam.read()  # 从摄像头读来的帧
        self.model = cv2.imread(fn)

        self.cam_work = 1
        h, w = prev.shape[:2]

        detect_time = 0


        # 根据监控摄像头判断当前的正面 侧面等

        self.frame_a = prev
        self.frame_b = prev
        self.frame_c = prev
        self.cnt = 0
        self.video_hit =va.VideoHits

    def detect(self):
        ret, frame = self.cam.read()

        if not ret:
            if self.DisplayMode:    # 历史视频判读
                self.cam = VideoCapture(self.fn)
                ret, img = self.cam.read()
            else:   # 读取视频失败
                # self.cam = VideoCapture(self.fn, self.num)
                # ret, img = self.cam.read()
                self.cam_work = 0
                print('读取摄像头{}视频失败'.format(self.num))

        self.frame_c = frame
        #self.frame_a,self.frame_b=shipin.chulishipin(self.model,self.frame_a,self.frame_b,self.frame_c)
        self.frame_a,self.frame_b,self.video_hit=ts.video_process(self.model,self.frame_a,self.frame_b,self.frame_c,self.video_hit)
        if len(self.video_hit)==3:
            self.video_hit.final_pre(24)
            self.video_hit=va.VideoHits  # 重新初始化




