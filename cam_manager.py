"""
该文件下完成cam的管理，负责控制信息的获取，命令的获取
"""
import tool
import threading
import time
import camcontrol as CAM
import cv2
import json
import paho.mqtt.client as mqtt

def on_connect(client,userdata,flags,rc):
    print("连接成功")
    client.subscribe("test")
    camera.start_init(0, DisplayMode, path, save_path, model, bullseye, innerdist, num_target)

def on_message(client,userdata,msg):
    payload = json.loads(msg.payload.decode())
    print("收到比赛ID："+payload.get("taskId"))
    tid = payload.get("taskId")
    camera.start(tid,client)









class CameraManager(object):
    def __init__(self, DisplayMode,cam_num=1):
        '''
        :param capture: 摄像头对象
        :param windowManager: 钩子类,窗口管理,按键
        '''
        # 从配置文件中读取截图目录和录像目录创建文件夹

        # 当前画面
        self._frame = None
        # 是否工作
        self._isWorking = True
        self.DisplayMode = DisplayMode
        self.cam_num = cam_num
        self.cam_list = [None]*self.cam_num
        self.framethread = [None]*cam_num

        tool.PRINT("视频采集器初始化完毕!")

    def start_init(self,idx,DisplayMode,path,save_path,model,bullseye,innerdist,num_target):
        # 开始工作
        tool.PRINT("开启视频采集")
        if self.DisplayMode:
            start = time.time()
            for idx in range(self.cam_num):
                self.cam_list[idx] = CAM.cam(idx, DisplayMode, path, save_path, model, bullseye, innerdist, num_target)

        else:
            for idx in range(self.cam_num):
                self.cam_list[idx] = CAM.cam(idx, DisplayMode, path, save_path, model, bullseye, innerdist, num_target)
        return self

    def start(self,tid,client):
        if self.DisplayMode:
            for idx in range(self.cam_num):
                framethread = threading.Thread(target=self.worker2, args=(self.cam_list[idx],tid,client))
                self.framethread[idx] = framethread
        else:
            for idx in range(self.cam_num):
                framethread = threading.Thread(target=self.worker, args=(self.cam_list[idx],tid,client))
                self.framethread[idx] = framethread
        for idx in range(self.cam_num):
            self.framethread[idx].start()




    def worker(self, ele,tid,client):
        print('3214')
        status = 0
        t1 = time.time()
        while self._isWorking:
            start = time.time()
            # try:
            if (start-t1)>19.5:
                tool.PRINT("未检测到箭")
                break
            normal,t1,status = ele.cam_detect(t1,tid,status,client)
            if not normal:
                tool.PRINT("识别完成")
                break
            while time.time() - start < 0.08:
                time.sleep(0.001)

    def worker2(self, ele,tid,client):
        print('12374')
        idx = 0
        t1 = time.time()
        status = 0
        while self._isWorking:
            start = time.time()
            # try:
            if idx == 15:
                idx = 0
            else:
                idx += 1
            normal,t1,status = ele.video_detect(idx,t1,tid,status,client)

            if not normal:
                tool.PRINT("识别完成")
                break
            while time.time() - start < 0.08:
                time.sleep(0.001)



if __name__ == "__main__":
    model = cv2.imread("input_video/model1.jpg")
    # model = cv2.imread("model1.jpg")
    # 详细信息
    model_h,model_w,_ = model.shape
    #point_x = int(model_h/2)
    #point_y = int(model_w/2)
    #bullseye = (point_x,point_y)
    bullseye = (378, 386)  # 靶心坐标
    innerdist = 35  # 环内径
    num_target = 10  # 环数


    num = 1  # 摄像头数目
    DisplayMode = 1  # 离线||摄像头
    path = "input_video/9_2.mp4"  # 离线视频地址
    # path = "120.avi"  # 离线视频地址
    save_path = "output"

    #video = cv2.VideoCapture(0)
    camera = CameraManager(DisplayMode)
    TASK_TOPIC = 'test'
    client_id = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

    client = mqtt.Client(client_id, transport='tcp')
    client.connect("127.0.0.1", 1883)

    client.on_connect = on_connect
    client.on_message = on_message

    client.loop_forever()

    """
    while True:
        frame = camera.getFrame()
        if frame is not None:
            cv2.imshow("test", frame)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                camera.stop()
                break
            elif k == 9:
                # tab键开启录像
                if not camera.isWritingVideo():
                    camera.startWritingVideo()
                else:
                    camera.stopWritingVideo()
            elif k == 32:
                # 空格键截图
                camera.writeImage()
    """
