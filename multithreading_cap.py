import time
import multiprocessing as mp
import cv2
import serial

import struct
import re
import config
import threading


global_dist = 0
global_speedx = 0
global_speedy = 0 # if speedy >0 move forward else backward
global_speedr = 0 # if speedr >0 turn left else turn right


def gstreamer_pipeline (capture_width=3280, capture_height=2464, display_width=480, display_height=360, framerate=21, flip_method=0) :
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))


def image_put(q):
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print('Successfully open Cam')
    else:
        print('Cam Open Failed')
    while True:
        q.put(cap.read()[1])
        q.get() if q.qsize() > 1 else time.sleep(0.01)


def image_get(q, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    while True:
        frame = q.get()
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)


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
            # print("speedx = %s, speedy = %s, speedr = %s \n"%(global_speedx,global_speedy,global_speedr))
        # return n

    def read_data(self):
        global global_dist
        while True:
            # self.message = self.port.readline()
            self.Bytedist = self.port.read(13)
            str_dist = str(self.Bytedist, encoding="utf-8")
            self.message = int(re.findall(r'\d+', str_dist)[0])
            global_dist = self.message
            print(self.message)


def tast_ult():
    print("testing")
    mSerial = SerialPort(config.serialPort, config.baudRate)
    try:
        t1 = threading.Thread(target=mSerial.read_data)
        t2 = threading.Thread(target=mSerial.set_speed)
    except:
        print("error")
    try:
        t1.start()
        t2.start()
        while True:
            mSerial.get_dist()
            print(global_dist)
    except:
        print("error")

def run_single_camera():
    mp.set_start_method(method='spawn')  # init
    queue = mp.Queue(maxsize=3)
    processes = [mp.Process(target=image_put, args=(queue, )),
                 mp.Process(target=image_get, args=(queue, 'Cam0')),
                 mp.Process(target=tast_ult)]

    [process.start() for process in processes]
    [process.join() for process in processes]



def run():
    run_single_camera()  # quick, with 2 threads
    pass


if __name__ == '__main__':
    run()