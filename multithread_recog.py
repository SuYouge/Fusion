import time
import multiprocessing as mp
import cv2
import serial
import post_process
import struct
import re
import config
import threading
import ctypes

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

global_dist = 0
global_speedx = 0
global_speedy = 0 # if speedy >0 move forward else backward
global_speedr = 0 # if speedr >0 turn left else turn right


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
            print('这是进程{0},线程{1}'.format(mp.current_process(), threading.current_thread()))
            print('Flag')


def Serial_threading():
    print("testing")
    try:
        mSerial = SerialPort(config.serialPort, config.baudRate)
        t1 = threading.Thread(target=mSerial.read_data)
        t2 = threading.Thread(target=mSerial.set_speed)
    except Exception as e:
        print(Exception, ": in Serial threading ", e)
    try:
        t1.start()
        t2.start()
        while True:
            mSerial.get_dist()
            print("distance is %s"%global_dist)
    except Exception as e:
            print(Exception, ": in start get dist ", e)


def get_recbox(x, img,label=None):
    list_1 = [0, [0, 0], [0, 0], 0]
    list_2 = [0, [0, 0], [0, 0], 0]
    # Plots one bounding box on image img
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    list = [list_1, list_2]
    name_list = ['balloon', 'ball']
    box = img[c1[1]:c2[1], c1[0]:c2[0]]
    if label:
        name = ''.join(re.findall(r'[A-Za-z]', label))
        posbility = ''.join(re.findall(r"\d+\.?\d*", label))
        # normalize coord shape[h,w]
        c1n, c2n = ((floatn(x[0] / img.shape[1]), floatn(x[1] / img.shape[0])),
                    (floatn(x[2] / img.shape[1]), floatn(x[3] / img.shape[0])))
        list[name_list.index(name)][0] = 1
        list[name_list.index(name)][1] = c1n
        list[name_list.index(name)][2] = c2n
        list[name_list.index(name)][3] = float(posbility)
        # print("label is %s"%label)
    return list, box

def gstreamer_pipeline (capture_width=3280, capture_height=2464, display_width=480, display_height=360, framerate=21, flip_method=0) :
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))


def set_size(im0,half):
    img, *_ = letterbox(im0, 416)
    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float16 if half else np.float32)  # uint8 to fp16/fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return img


def image_put(q,):
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print('Successfully open Cam')
    else:
        print('Cam Open Failed')
    try:
        while True:
            q.put(cap.read()[1])
            q.get() if q.qsize() > 1 else time.sleep(0.01)
    except Exception as e:
        print(Exception, ": in image put ", e)


def image_get(q,device, model, classes, colors):
    cv2.namedWindow('Cam0', flags=cv2.WINDOW_FREERATIO)
    half = True and device.type != 'cpu'
    try:
        while True:
            im0 = q.get()
            img = set_size(im0, half)
            print("in recognition")
            img = torch.from_numpy(img).unsqueeze(0).to(device)
            pred, _ = model(img)
            det = non_max_suppression(pred.float(), config.conf_thres, config.nms_thres)[0]
            s = '%gx%g ' % img.shape[2:]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # add to string
                for *xyxy, conf, _, cls in det:
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    print("label is %s" % label)
                    list, box = get_recbox(xyxy, im0, label=label)
                    # cache.put(list) # if get immediately no more list in cache
                    if (classes[int(cls)] == 'balloon'):
                        cntm, color = post_process.get_color(box)
                        if (cntm is not None):
                            match = post_process.match_img(im0, box, 0.8)
                            cv2.rectangle(im0, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 1)
            cv2.imshow('Cam0', im0)
            cv2.waitKey(1)
    except Exception as e:
        print(Exception, ": in image get ", e)


def check_color(box,target_color):
    try:
        _,color = post_process.get_color(box)
    except:
        return 0
    else:
        if (color == target_color):
            return 1
        else:
            return 0


def init_detect():
    img_size = 416
    device = torch_utils.select_device(force_cpu=ONNX_EXPORT)
    torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
    model = Darknet(config.cfg, img_size)
    model.load_state_dict(torch.load(config.weights, map_location=device)['model'])
    model.to(device).eval()
    half = True and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()
    classes = load_classes(parse_data_cfg(config.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    return device,model,classes,colors


def run_single_camera():
    device, model, classes, colors = init_detect()
    mp.set_start_method(method='spawn')  # init
    img_queue = mp.Queue()
    dist_queue = mp.Queue()
    # track_cnt_queue = mp.Queue(maxsize=3)

    processes = [
                 # mp.Process(target=image_put, args=(img_queue,)),
                 # # mp.Process(target=torch_recog_threading,
                 # #            args=(img_queue,device, model, classes, colors))
                 # mp.Process(target=image_get, args=(img_queue,device, model, classes, colors)),
                 mp.Process(target=Serial_threading)
                 ]

    [process.start() for process in processes]
    [process.join() for process in processes]



def run():
    run_single_camera()  #
    pass


if __name__ == '__main__':
    run()