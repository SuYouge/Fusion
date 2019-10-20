import serial
import struct
import re
import threading
import time
import config
# test = serial.Serial("COM14", 9600,bytesize=8, parity='N',timeout=200)
global_flag = 1
target_mode = 'PS'
# target_mode = 'BT'

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
    def set_mode(self):
        self.package = struct.pack('<3s3hs', b'\xff\xdd\x04', 0, 0, 0, b'\x00')
        self.port.write(self.package)
    def read_data(self):
        global global_flag
        while True:
            # self.message = self.port.readline()
            self.Bytedist = self.port.read(13)
            str_dist = str(self.Bytedist, encoding="utf-8")
            # self.message = int(re.findall(r'\d+', str_dist)[0])
            # global_dist = self.message
            if (str_dist[2:4]== target_mode):
                global_flag = 0
            print(str_dist[2:4])

if __name__ == '__main__':
    mSerial = SerialPort(config.serialPort, config.baudRate)
    t1 = threading.Thread(target=mSerial.read_data)
    t1.start()
    # t2 = threading.Thread(target=mSerial.set_mode)
    # t2.start()
    while global_flag:
        mSerial.set_mode()
        time.sleep(0.1)

