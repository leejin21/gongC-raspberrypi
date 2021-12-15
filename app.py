import requests, json
from requests.api import post
import torch
import numpy as np
import cv2
import json
import time

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

def getResultArray(results):
    '''
    [OUTPUT 형태]
    [dict 객체, dict 객체, ...]
    예시: [{'xmin': 226.7082824707, 'ymin': 210.1504211426, 'xmax': 561.6719970703, 'ymax': 480.0, 'confidence': 0.5927491188, 'class': 0, 'name': 'handonly'}, {'xmin': 300.7040100098, 'ymin': 30.6222915649, 'xmax': 479.7358703613, 'ymax': 270.3709106445, 'confidence': 0.2770100236, 'class': 9, 'name': 'phone'}]
    '''
    results_array = results.pandas().xyxy[0].to_json(orient="records")      # 결과값 json변환
    results_array = results_array.replace("'","\"")     # '을 "로 치환해야 json으로 변환 가능함
    results_array = json.loads(results_array)       # string을 json(dict)형식으로 변환
    return results_array

def postDataBy1Min(data):
    URL = 'http://localhost:5000/concent/data'
    headers = {'Content-Type': 'application/json; charset=utf-8', 'x-access-token':'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6InRlc3RAZ21haWwuY29tIiwiaWF0IjoxNjM3NjU4NjY0LCJleHAiOjUyMzc2NTUwNjR9.E2MQKePtt5A21XvVsrYi2kPxbc-M25b6m7dS-NErfsk'}
    cookies = {'session_id': 'sorryidontcare'}
    res = requests.post(URL, data, headers=headers, cookies=cookies)
    print("*"*50)
    if res.status_code == 200:
        # 성공 시
        print("POST /concent/data 성공")
        print("*"*50)
        return True
    else:
        print("POST /concent/data 실패")
        print("*"*50)
        return False

def isConcentOrPlay(data):
    '''
    [OUTPUT]: 'C' or 'P'
    1. C인 경우
        손, 책이 둘 다 검출되는 경우
        손, 태블릿이 둘 다 검출되는 경우
        손, 펜이 둘 다 검출되는 경우 
    2. P인 경우
        핸드폰인 경우
        C가 아닌 경우
    '''
    isHandExist = False
    isBookExist = False
    isPenExist = False
    
    for d in data:
        if d['name'] == 'handonly':
            isHandExist = True
        if d['name'] == 'book' or d['name'] == 'tablet':
            isBookExist = True
        if d['name'] == 'pen':
            isPenExist = True
        if d['name'] == 'phone':
            return 'P'

    if isHandExist and isBookExist:
        return 'C'
    elif isHandExist and isPenExist:
        return 'C'
    return 'P'
        
    
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
    player = cv2.VideoCapture(0)
    assert player.isOpened()
    # 0 means read from local camera
    current_time = {"hour": time.localtime().tm_hour, "minute": str(time.localtime().tm_min)}
    study_data = {"C": 0, "P":0}

    while True:
        prev_minute = current_time["minute"]

        start_time = time.time()
        ret, frame = player.read()
        assert ret
        results = model(frame)
        end_time = time.time()

        fps = 1/np.round(end_time - start_time, 3)
        
        # 분이 바뀔 때마다
        current_time = {"hour": time.localtime().tm_hour, "minute": str(time.localtime().tm_min)}
        if (prev_minute != current_time["minute"]):
            if study_data["C"] > study_data["P"]:
                # 1분간 집중한 경우로 취급
                body = {"status": "C"}
            else:
                # 1분간 논 경우로 취급
                body = {"status": "P"}
            # 서버에 POST 요청
            if postDataBy1Min(body):
                # 성공적으로 post 한 경우 클리어해 주기
                study_data = {"C": 0, "P":0}
            
        # results -> results_array(요소 각각이 딕셔너리인 배열)로 변환
        results_array = getResultArray(results)
        # 공부 여부 판별 알고리즘 적용해 study_data에 C, P 개수 갱신 및 축적
        study_data[isConcentOrPlay(results_array)] += 1

        # 로그 찍기
        print("===============================")
        print(f"Frame Per Second: {fps}")
        print("Current Time: "+str(current_time["hour"])+":"+str(current_time["minute"])+":"+str(time.localtime().tm_sec))
        for result in results_array:
            print(result["name"], end=" ")
            print(round(result["confidence"], 3), end=" ")
                # 소수점 아래 셋째 자리까지 반올림
            print("")
        print(results_array)

if __name__== "__main__":
    # 해당 파일 실행 시 실행

    model = torch.hub.load('yolov5', 'custom', path='weights/best.pt', source='local', force_reload=True)
    
    main()
    # postDataBy1Min({})
