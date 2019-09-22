import serial
import time
import threading
import struct
import re

global_dist = 0
global_speedx = 0
global_speedy = 0
global_speedr = 0

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
        self.package = struct.pack('<3s3hs', b'\xff\xfe\x01', global_speedx, global_speedy,global_speedr, b'\x00')
        n = self.port.write(self.package)
        return n

    def read_data(self):
        global global_dist
        while True:
            # self.message = self.port.readline()
            self.Bytedist = self.port.read(13)
            str_dist = str(self.Bytedist, encoding="utf-8")
            self.message = int(re.findall(r'\d+', str_dist)[0])
            global_dist = self.message
            # print(self.message)



serialPort = "COM14"  # 串口
baudRate = 9600  # 波特率

if __name__ == '__main__':

    mSerial = SerialPort(serialPort, baudRate)
    t1 = threading.Thread(target=mSerial.read_data)
    t1.start()
    while True:
        # time.sleep(1)
        mSerial.get_dist()
        print(global_dist)
        global_speedx = 700
        global_speedy = 0
        global_speedr = 0
        mSerial.set_speed()