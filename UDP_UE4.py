import cv2
from cvzone.ColorModule import ColorFinder
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time
import mediapipe as mp
# import autopy
import DAFFilter
import socket


wScr, hScr = 1920, 1080   # 返回电脑屏幕的宽和高(1920.0, 1080.0)
wCam, hCam = 1920, 1080   # 视频显示窗口的宽和高
pt1, pt2 = (200,200), (1720, 880)   # 虚拟鼠标的移动范围，左上坐标pt1，右下坐标pt2
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 3002)

cap = cv2.VideoCapture(0)
cap.set(3, wCam)  # 设置显示框的宽度1280
cap.set(4, hCam)  # 设置显示框的高度720
ptime = 0
pLocx, pLocy = 0, 0  # 上一帧时的鼠标所在位置
smooth = 4  # 自定义平滑系数，让鼠标移动平缓一些
frames = 32
start = 0
end = 2 * np.pi
scale = 0.5
x_hat1,x_hat2,data2,data1=np.array([[0,0],[0,0]]),np.array([[0,0],[0,0]]),np.array([[0,0],[0,0]]),np.array([[0,0],[0,0]])
for j in range(4):
    data2 = np.append(data2,data2,axis=0)
    data1 = np.append(data1, data1, axis=0)
    x_hat2 =np.append(x_hat2, x_hat2, axis=0)
    x_hat1 =np.append(x_hat1, x_hat1, axis=0)

detector = HandDetector(maxHands=1, detectionCon=0.8,minTrackCon=0.5)



while True:
    success, img = cap.read()
    #img = cv2.flip(img,1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        hand1 = hands[0]
        #print(hand1["lmList"])

       # print(str.encode(str(hand1)))
        start0 = time.time()
        lmList1 = hand1["lmList"]
        x1, y1 ,z1= lmList1[4]  # 食指尖的关键点索引号为8
        temp = np.array([[x1, y1]])
        data1 = np.append(data1, temp, axis=0)
        data_temp1 = np.delete(data1, 0, 0)
        data1 = data_temp1
        t_1 = np.linspace(start, end, frames)
        beta = max(data_temp1) - min(data_temp1)
        DAF1 = DAFFilter.DAFFilter(t_1[0], data_temp1[0], min_cutoff=0.15, beta=beta)
        for i in range(2, len(t_1)):
            x_hat1[i] = DAF1.filter_signal(t_1[i], data_temp1[i])
            x_aver1 = np.mean(x_hat1, axis=0)
        x_aver1 = np.mean(x_hat1, axis=0)
        x1 = x_aver1[0]
        y1 = x_aver1[1]
        x2, y2 ,z2= lmList1[8]  # 中指指尖索引12
        
        # （5）检查哪个手指是朝上的
        fingers = detector.fingersUp(hands[0])  # 传入
        # print(fingers) 返回 [0,1,1,0,0] 代表 只有食指和中指竖起

        # 如果食指竖起且中指弯下，就认为是移动鼠标
        if fingers[0] == 1 and fingers[1] == 0 or fingers[2] == 0 or fingers[3]==0 :
            # 开始移动时，在食指指尖画一个圆圈，看得更清晰一些
            cv2.circle(img, (x2, y2), 15, (255, 255, 0), cv2.FILLED)  # 颜色填充整个圆

            # （6）确定鼠标移动的范围
            # 将食指的移动范围从预制的窗口范围，映射到电脑屏幕范围

            x3 = np.interp(x1, (pt1[0], pt2[0]), (0, wScr))
            y3 = np.interp(y1, (pt1[1], pt2[1]), (0, hScr))
            cLocx = pLocx + (int(x3) - pLocx) / smooth  # 当前的鼠标所在位置坐标
            cLocy = pLocy + (int(y3) - pLocy) / smooth
            # （7）移动鼠标
            if 0<x3<1920 and 0<y3<1080:
                # autopy.mouse.move(x3, y3)  # 给出鼠标移动位置坐标

                pLocx, pLocy = cLocx, cLocy
                x_raw = int((x3/1920)*360)
                y_raw = int((y3/1080)*360)
                lmList2 = [x_raw,y_raw]
                print(lmList2)
                end0 = time.time()
                print('Running time: %s Seconds' % (end0 - start0))
                sock.sendto(str.encode(str(lmList2)), serverAddressPort)



        #x1, y1 ,z1= lmList1[8][0], lmList1[8][1],lmList1[8][2]
        #cv2.circle(img,(x1,),10,(255,0,0),cv2.FILLED)
        #print(x1,y1,z1)y1
        #dat=  x1, y1,z1
      # print(str.encode(str(dat)))

        #sock.sendto(str.encode(str(lmList1)),serverAddressPort)


    ctime = time.time()
    fps = 1//(ctime-ptime)
    ptime = ctime

    cv2.rectangle(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 255, 255), 2)
    cv2.putText(img, f'FPS:{int(fps)}', (30, 30), cv2.FONT_ITALIC, 1, (0, 0, 200), 3)


    img = cv2.resize(img,(0,0),None,0.5,0.5)


    cv2.imshow("Image", img)
    cv2.waitKey(1)
