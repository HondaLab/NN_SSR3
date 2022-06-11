#!/usr/bin/python3

# collection_data.py
# Execute this on a robot.

import modules.keyin as keyin # キーボード入力を監視するモジュール
import modules.motor5a as mt # pwmでモーターを回転させるためのモジュール
import modules.imageCut as ic
import cv2
import time
import pigpio
from subprocess import Popen
import numpy as np
import modules.vl53_4a as lidar
import socket
import modules.socket as sk
import time
import modules.camera as camera

tofL,tofC,tofR=lidar.start3()
print(tofL,tofC,tofR)

picture_data = []

def Run(mL,mR,left,right):
    if left<-100: left = -99
    if left>100: left = 99
    if right<-100: right = -99
    if right>100: right = 99
    mL.run(left)
    mR.run(right)

def send_data(l,r,vw):
    camera.cam.capture(camera.rawCapture, format="bgr", use_video_port=True)
    frame = camera.rawCapture.array
    frame2 = frame[ic.im_cut_up:ic.im_cut_below,:,:]
    vw.write(frame2)
    cv2.imshow('frame',frame2)
    cv2.waitKey(1)
    for i in range(0,camera.RES_X):
       picture_data.append(sum(frame[ic.im_cut_up:ic.im_cut_below,i,0]))
    for i in range(0,camera.RES_X):
       picture_data.append(sum(frame[ic.im_cut_up:ic.im_cut_below,i,1]))
    for i in range(0,camera.RES_X):
       picture_data.append(sum(frame[ic.im_cut_up:ic.im_cut_below,i,2]))

    distanceL=tofL.get_distance()
    if distanceL>2000:
        distanceL=2000
    distanceC=tofC.get_distance()
    if distanceC>2000:
        distanceC=2000
    distanceR=tofR.get_distance()
    if distanceR>2000:
        distanceR=2000
    #print("\r %4d %4d" % (distanceL,distanceC,distanceR),end='')
    picture_data.append(distanceL)
    picture_data.append(distanceC)
    picture_data.append(distanceR)
    if l<-100: l = -99
    if l>100: l = 99
    if r<-100: r = -99
    if r>100: r = 99
    picture_data.append(l)
    picture_data.append(r)
    udp_send_data.send(picture_data)
    picture_data.clear()
    camera.rawCapture.truncate(0)

if __name__=="__main__":
    OUT_FILE="data_collection_output.mp4"
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    record_fps=10
    width=320
    height=ic.im_cut_below-ic.im_cut_up
    print("# Resolution: %5d x %5d" % (width,height))
    size = (width, height)
    vw = cv2.VideoWriter(OUT_FILE, fmt, record_fps, size)
    udp_send_data = sk.UDP_Send(sk.recv_addr,sk.sensor_port)
   
    STEP=20
    HANDLE_STEP=15
   
    right_flag = 0
    left_flag = 0

    mL=mt.Lmotor(17)
    mR=mt.Rmotor(18)
   
    key = keyin.Keyboard()
    ch="c"
    send_start = "stop"
    PERIOD=0.2
    now = time.time()
    start = now
    init = now
    print("Input q to stop.")
    left=0
    right=0
    send_num = 0
    while ch!="q":
        ch = key.read()
        print("\r %4d %4d" % (left,right),end='')
        try:
            if ch == "a" :
                left+= STEP
                right+= STEP
                send_start = "start"
                send_data(left,right,vw)
                #in_data_posi = in_data_posi + 1
                Run(mL,mR,left,right)
            if ch == "z" :
                left-= STEP
                right-= STEP
                send_data(left,right,vw)
                #in_data_posi = in_data_posi + 1
                Run(mL,mR,left,right)

            if ch == "j" :
                right = right + HANDLE_STEP
                left = left - HANDLE_STEP
                right_flag = right_flag + HANDLE_STEP
                left_flag = left_flag - HANDLE_STEP
                send_data(left,right,vw)
                #in_data_posi = in_data_posi + 1
                Run(mL,mR,left,right)
            if ch == "k" :
                right = right - right_flag
                left = left - left_flag
                right_flag = 0
                left_flag = 0
                send_data(left,right,vw)
                #in_data_posi = in_data_posi + 1
                Run(mL,mR,left,right)
            if ch == "l" :
                right = right - HANDLE_STEP
                left = left + HANDLE_STEP
                right_flag = right_flag - HANDLE_STEP
                left_flag = left_flag + HANDLE_STEP
                send_data(left,right,vw)
                #in_data_posi = in_data_posi + 1
                Run(mL,mR,left,right)

            '''    
            Run(mL,mR,left,right)
            if send_start == "start":
                if now-init > PERIOD:
                    send_data(left,right)
                    init = now
            now = time.time()
            '''
        except KeyboardInterrupt:
            mL.run(0)
            mR.run(0)
            break

    print("\nTidying up")
    vw.release()
    mL.run(0)
    mR.run(0)
   
