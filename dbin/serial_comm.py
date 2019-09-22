import serial
import struct
import re
import threading
import time
# test = serial.Serial("COM14", 9600,bytesize=8, parity='N',timeout=200)
serialPort = "COM14"  # 串口
baudRate = 9600  # 波特率

# MyThread.py线程类

# class MyThread(threading.Thread):
#
#     def __init__(self, func, args=()):
#         super(MyThread, self).__init__()
#         self.func = func
#         self.args = args
#
#     def run(self):
#         # time.sleep(0.02)
#         self.result = self.func(*self.args)
#
#     def get_result(self):
#         threading.Thread.join(self)  # 等待线程执行完毕
#         try:
#             return self.result
#         except Exception:
#             return None


class SerialPort:
    message = ''
    def __init__(self, port, buand):
        super(SerialPort, self).__init__()
        self.port = serial.Serial(port, buand,bytesize=8, parity='N',timeout=200)
        self.port.close()
        if not self.port.isOpen():
            self.port.open()

    def port_open(self):
        if not self.port.isOpen():
            self.port.open()

    def port_close(self):
        self.port.close()

    def get_dist(self):
        package = struct.pack('<3s3hs', b'\xff\xee\x02', 0, 0, 0, b'\x00')
        n = self.port.write(package)
        dist = self.read_data()
        return dist

    def set_speed(self,speedx,speedy,speedr):
        package = struct.pack('<3s3hs', b'\xff\xfe\x01', speedx, speedy,speedr, b'\x00')
        n = self.port.write(package)
        return n

    def read_data(self):
        while True:
            # self.get_dist()
            self.Bytedist = self.port.read(13)
            str_dist = str(self.Bytedist, encoding="utf-8")
            self.message = int(re.findall(r'\d+', str_dist)[0])
            print("message is %s"%self.message)
            # return self.Bytedist


if __name__ == '__main__':
    mSerial = SerialPort(serialPort, baudRate)
    task = threading.Thread(target=mSerial.read_data)
    task.start()
    # print("task is alive %s" % task.is_alive())
    while True:
        # print("task is alive %s" % task.is_alive())
        mSerial.get_dist()
        # print(task.get_result())
        # mSerial.set_speed(100,0,0)



