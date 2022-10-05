# 订阅-发送端测试
import json
import sys
import os
import paho.mqtt.client as mqtt
import time


def on_connect(client,userdata,flags,rc):
    print("连接成功")

    client.subscribe("targetscore")
    print("连接"+"targetscore")
    client.publish("test", json.dumps({"taskId": "3-1"}))

def on_message(client,userdata,msg):
    payload = json.loads(msg.payload.decode())

    if (payload.get("status")==0):
        print("已收到箭")
    elif (payload.get("status")==1):
        print("taskID:"+payload.get("taskId")+"\tscore:"+str(payload.get("score")))
        client.publish("test", json.dumps({"taskId": "3-1"}))



if __name__ == '__main__':
    TASK_TOPIC = 'test'
    client_id = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

    client = mqtt.Client(client_id, transport='tcp')
    client.connect("127.0.0.1", 1883)

    client.on_connect = on_connect
    client.on_message = on_message

    client.loop_forever()
