# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import ImgProcess as pro
import numpy as np
from fastapi import FastAPI
from flask import Flask

def add(a,b):
    return a+b


app = FastAPI()
@app.get("/")
def index():
    return {"msg": "hello,world"}

res = add(1,2)  # 本地调用
print(res)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# RPC 一个接口的请求过程
@app.get("/add_by_get")
def add_get():
    c = add(1,2)
    return {"c":c}