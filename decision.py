import time
import multiprocessing as mp
import serial
import post_process
import struct
import config
import threading
import ctypes
import re
import cv2

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

global_speedx = 0
global_speedy = 0 # if speedy >0 move forward else backward
global_speedr = 0 # if speedr >0 turn left else turn right


cache_box = []
cache_size = 5
shake_flag = 0
shake_cnt = 0
diappear_flag = -1



