# 订阅端测试
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
    print("taskId："+payload.get("taskId") )
    client.publish("test", json.dumps({"taskId": "3-1"}))



if __name__ == '__main__':
    TASK_TOPIC = 'test'
    client_id = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

    client = mqtt.Client(client_id, transport='tcp')
    client.connect("127.0.0.1", 1883)

    client.on_connect = on_connect
    client.on_message = on_message

    client.loop_start()


    while True:
        continue