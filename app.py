import requests
import torch
import numpy as np
import cv2
import json

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def test():
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Image
    img = 'https://ultralytics.com/images/zidane.jpg'

    # Inference
    results = model(img)

    print(results.pandas().xyxy[0])


def main():
    global model
    '''
    [기능]
    1. 5초마다 캠으로 집중 여부 파악하기
        (1) DUMMY: 로그로 표시(객체 -> 객체 간 근접 여부 -> 집중 여부)
        (2) REAL: 집중 여부 LED로 표시
    2. 60초마다 축적한 12개의 집중 여부 -> 서버로 전송
    3. (확장) 전광판 사용하기
    '''
    stream = cv2.VideoCapture(0)
    # 0 means read from local camera
    results = model(stream)
    results.print()
    pass


if __name__== "__main__":
    # 해당 파일 실행 시 실행

    model = torch.hub.load('gongC-raspberrypi/yolov5', 'custom', path='gongC-raspberrypi/weights/best.pt', source='local', force_reload=True)
    
    # main()
    pass
