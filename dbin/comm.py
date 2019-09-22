import serial
import struct
import re
import os
import sys
import time
test = serial.Serial("COM14", 9600,bytesize=8, parity='N',timeout=200)

# t_head = b'\xee\xfe\x01'
# head = b'\xff\xfe\x01'
# tail = b'\x00'

# while True:
#
#     speedx =700
#     speedy = 0
#     speedr = 0
#     package = struct.pack('<3s3hs',t_head,speedx,speedy,speedr,tail)
#     n = fd.write(package)
#     time.sleep(0.5)
#     msg = fd.read(13)
#     dist = str(msg, encoding="utf-8")
#     # msg_tx = struct.unpack('<3s3hs',msg)
#     # print(msg)
#     print(dist)

# class dist:
#     def __init__(self):
#         self.name = None
#         print("%s\n"%self.name)
#
#     def read(self,fd):
#         package = struct.pack('<3s3hs', t_head, 0, 0, 0, tail)
#         n = fd.write(package)
#         dist = fd.read(13)
#         dist = str(dist, encoding="utf-8")
#         return dist
#
#     def __del__(self):
#         print("%s destroied"%self.name)


def distRead(stamp,fd):
    # print(fd.is_open)
    if (fd.isOpen() == False):
        fd.open()
    # print(fd.is_open)
    package = struct.pack('<3s3hs', b'\xff\xee\x02', 0, 0, 0, b'\x00')
    # package = b'\xff \xee \x02 \x00 \x00 \x00 \x00 \x00 \x00 \x00'
    try:
        cnt = fd.write(package)

        print("write")
        time.sleep(0.02)
        # print(cnt)
        # fd.flush()
    except Exception as e:
        print("---Error---：", e)
    try:
        Bytedist = fd.read(1) # read 阻塞 读不到数据
        print("read")
        # if (Bytedist == None):
        #     _ = fd.write(package)
        # Bytedist = fd.read(13)
    except Exception as e:
        print("---Error---：", e)
    str_dist = str(Bytedist, encoding="utf-8")
    dist = int(re.findall(r'\d+', str_dist)[0])
    # print("%sth pic"%stamp)
    fd.close()
    return dist

def speedWrite(stamp,fd,speedx,speedy,speedr):
    if (fd.isOpen() == False):
        fd.open()
    package = struct.pack('<3s3hs', b'\xff\xfe\x01', speedx, speedy, speedr, b'\x00')
    try:
        # Bytedist = fd.read()
        _ = fd.write(package)
        # fd.read()
        # fd.flushOutput()
    except Exception as e:
        print("---Error---：", e)
    fd.close()
    print("%sth pic write speed %s %s %s "%(stamp,speedx,speedy,speedr))



if __name__ == '__main__':
    while True:
        t_dist = distRead(0,test)
        print(t_dist)
        # speedWrite(0,test,100,0,0)
        # test.close()
        # package = b'\xff\xee\x02\x00\x00\x00\x00\x00\x00\x00\xff\xfe\x01\x00\x00\x00\x00\x00\xff\x00'
        # cnt = test.write(package)
        # print(cnt)
        # Bytedist = test.read(13)
        # str_dist = str(Bytedist, encoding="utf-8")
        # dist = int(re.findall(r'\d+', str_dist)[0])
        # print(dist)