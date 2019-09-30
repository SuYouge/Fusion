import cv2
import numpy as np
import random
import serial
import struct
import re
import threading
import copy
import time

global_dist = 0
global_speedx = 0
global_speedy = 0 # if speedy >0 move forward else backward
global_speedr = 0 # if speedr >0 turn left else turn right
global_decpos = [[0,0,0,0],[0,0,0,0]] # [obj1[x_upleft,x_downright], obj2[x_upleft,x_downright]]
global_imgsize = [0,0] # [width,height]
global_dist_thresh = 200
serialPort = "COM14"  # serial no
baudRate = 9600  # Baudrate
box_num = 1
test_mode = True

length_thresh = 0.05

cache_box = []

def suppress(speed):
    if (speed>1000):
        speed = 1000
    elif(speed<-1000):
        speed = -1000
    # x = speed
    # speed = 1 / (1 + np.exp(-x))
    return speed


# serial port operation
class SerialPort:
    message = ''

    def __init__(self, port, buand):
        super(SerialPort, self).__init__()
        self.port = serial.Serial(port, buand)
        self.port.close()
        if not self.port.isOpen():
            self.port.open()

    def port_open(self):
        if not self.port.isOpen():
            self.port.open()

    def port_close(self):
        self.port.close()

    def get_dist(self):
        # data = input("请输入要发送的数据（非中文）并同时接收数据: ")
        # n = self.port.write((data + '\n').encode())
        self.package = struct.pack('<3s3hs', b'\xff\xee\x02', 0, 0, 0, b'\x00')
        n = self.port.write(self.package)
        return n

    def set_speed(self):
        global global_speedx
        global global_speedy
        global global_speedr
        # self.n=0
        # while (self.n%100 == 0):
        while True:
            # self.n+=1
            self.package = struct.pack('<3s3hs', b'\xff\xfe\x01', global_speedx, global_speedy,global_speedr, b'\x00')
            n = self.port.write(self.package)
            print(global_speedx,global_speedy,global_speedr)
        # return n

    def read_data(self):
        global global_dist
        while True:
            # self.message = self.port.readline()
            self.Bytedist = self.port.read(13)
            str_dist = str(self.Bytedist, encoding="utf-8")
            self.message = int(re.findall(r'\d+', str_dist)[0])
            global_dist = self.message
            # print(self.message)


def model():
    global global_speedx
    global global_speedy
    global global_speedr
    global_speedx =0
    global_speedy = 100
    global_speedr = 0


def find_box(frame):
    box = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gradX = cv2.Sobel(gray, cv2.CV_32F, 1, 0, -1)
    gradY = cv2.Sobel(gray, cv2.CV_32F, 0, 1, -1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 80, 160, cv2.THRESH_BINARY) ### 160，160
    cv2.imshow('thresh', thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7)) # 21,7
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=16) # 4
    cv2.imshow('closed', closed)
    closed = cv2.dilate(closed, None, iterations=8) # 4

    cnts, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts)>box_num:
        for index in range(box_num):
            c = sorted(cnts, key=cv2.contourArea, reverse=True)[index]
            rect = cv2.minAreaRect(c)
            #
            box_list = np.int0(cv2.boxPoints(rect))
            centerx = (box_list[0][0] +box_list[2][0])/2
            centery = (box_list[0][1] + box_list[2][1]) / 2
            length = ((((box_list[0][0] - box_list[1][0]))**2 +(box_list[0][1] - box_list[1][1])**2)**0.5)*0.5
            length = calength(box_list)
            L_up = (int(centerx-0.5*length),int(centery-0.5*length))
            R_down = ((int(centerx+0.5*length),int(centery+0.5*length)))
            # print(L_up,R_down)
            # print(box_list)
            box_list = (L_up,R_down)
            box.append(box_list)
            #
    elif len(cnts)<=box_num:
        for index in range(len(cnts)):
            c = sorted(cnts, key=cv2.contourArea, reverse=True)[index]
            rect = cv2.minAreaRect(c)
            #
            box_list = np.int0(cv2.boxPoints(rect))
            centerx = (box_list[0][0] +box_list[2][0])/2
            centery = (box_list[0][1] + box_list[2][1]) / 2
            # length = (((box_list[0][0] - box_list[1][0]))**2 +(box_list[0][1] - box_list[1][1])**2)**0.5
            length = calength(box_list)
            L_up = (int(centerx-0.5*length),int(centery-0.5*length))
            R_down = ((int(centerx+0.5*length),int(centery+0.5*length)))
            # print(L_up,R_down)
            # print(box_list)
            box_list = (L_up,R_down)
            box.append(box_list)
            #
    # nor_box = normalize(box, frame.shape)
    box = box_filter(box, frame)
    return box

def box_filter(box,frame):
    nor_box = normalize(box, frame.shape)
    if len(nor_box):
        for i in range(len(nor_box)):
            l = calength(nor_box[i])
            if l < length_thresh:
                box.pop(i)
    return box

def calength(box):
    length = (((box[0][0] - box[1][0])) ** 2 + (box[0][1] - box[1][1]) ** 2) ** 0.5
    return length

def gstreamer_pipeline(capture_width=3280, capture_height=2464, display_width=480, display_height=360, framerate=21,
                       flip_method=0):
    return ('nvarguscamerasrc ! '
            'video/x-raw(memory:NVMM), '
            'width=(int)%d, height=(int)%d, '
            'format=(string)NV12, framerate=(fraction)%d/1 ! '
            'nvvidconv flip-method=%d ! '
            'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
            'videoconvert ! '
            'video/x-raw, format=(string)BGR ! appsink' % (
            capture_width, capture_height, framerate, flip_method, display_width, display_height))


# return normalized box position
def normalize(box,img_size):
    norm_box = []
    lx,ly,rx,ry = 0,0,0,0
    if len(box):
        for index in range(len(box)):
            lx = box[index][0][0]/img_size[0]
            ly = box[index][0][1] / img_size[1]
            rx = box[index][1][0] / img_size[0]
            ry= box[index][1][1] / img_size[1]
            norm_box.append([(lx,ly),(rx,ry)])
    # print(norm_box)
    return norm_box


def detect_qrcode():
    # cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(0)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(3)]
    while (cap.isOpened()):
        ret, frame = cap.read()
        ###
        if (test_mode is not True):
            mSerial.get_dist()
            print(global_dist)
            model()
        ###
        # cv2.flip(frame, 0, frame) # uncomment in jetson nano
        if ret:
            box = find_box(frame)
            # nor_box = normalize(box, frame.shape)
            if len(box)>box_num:
                for index in range (box_num):
                    # print("box %s is at\n"%index)
                    # print(box[index])
                    cv2.rectangle(frame, box[index][0], box[index][1], colors[index], 3)
                    # cv2.drawContours(frame, [box[index]], 0, colors[index], 3)
            elif len(box)<=box_num:
                for index in range (len(box)):
                    # print("box %s is at\n"%index)
                    # print(box[index])
                    # cv2.drawContours(frame, [box[index]], 0, colors[index], 3)
                    cv2.rectangle(frame, box[index][0], box[index][1], colors[index], 3)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ###
    if (test_mode is not True):
        mSerial = SerialPort(serialPort, baudRate)
        t1 = threading.Thread(target=mSerial.read_data)
        t1.start()
        t2 = threading.Thread(target=mSerial.set_speed)
        t2.start()
    ###
    detect_qrcode()
