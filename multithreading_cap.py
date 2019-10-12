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


def image_put(q):
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print('Successfully open Cam')
    else:
        print('Cam Open Failed')
    try:
        while True:
            q.put(cap.read()[1])
            q.get() if q.qsize() > 4 else time.sleep(0.01)
    except Exception as e:
        print(Exception, ": in image put ", e)


def image_get(q, boxq, window_name):
    cv2.namedWindow(window_name, flags=cv2.WINDOW_FREERATIO)
    try:
        while True:
            frame = q.get()
            if (boxq.qsize()>0):
                box = boxq.get(block = False)
                cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),(255,0,0),1)
            cv2.imshow(window_name, frame)
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


def torch_recog_threading(imgq,boxq,initbox,device, model, classes, colors,color_queue,recondone_queue):
    try :
        half = True and device.type != 'cpu'
        with torch.no_grad():
            while True:
                # if ((recondone_queue.value != 1)):
                im0 = imgq.get()
                img = set_size(im0, half)
                print("in recognition")
                img = torch.from_numpy(img).unsqueeze(0).to(device)
                pred, _ = model(img)
                # non_max_suppression (x1, y1, x2, y2, object_conf, class_conf, class)
                det = non_max_suppression(pred.float(), config.conf_thres, config.nms_thres)[0]
                # s = '%gx%g ' % img.shape[2:]
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # s += '%g %ss, ' % (n, classes[int(c)])  # add to string
                    for *xyxy, conf, _, cls in det:
                        label = '%s %.2f' % (classes[int(cls)], conf)
                        # print("label is %s"%label)
                        list, box = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                        # cache.put(list) # if get immediately no more list in cache
                        if (classes[int(cls)] == 'balloon'):
                            cntm, color = post_process.get_color(box)
                            color_queue.value = bytes(color, encoding = "utf8")
                            # print(color_queue.value)
                            if (cntm is not None):
                                match = post_process.match_img(im0, box, 0.8)
                        #         temp_cache = box
                        #         print("flag set")
                                initbox.put(xyxy)
                                initbox.get() if initbox.qsize() > 1 else time.sleep(0.01)
                                # recondone_queue.value = 1
                                boxq.put(xyxy)
                                boxq.get() if boxq.qsize() > 1 else time.sleep(0.01)
                    #         # cv2.imshow("box", box)
    except Exception as e:
        print(Exception, ": in recognizing ", e)


def tracking_thread(initbox,imq,boxq,color_queue,track_cnt,recondone_queue):
    try:
        init_flag = 0
        print("once")
        while (imq.qsize()):
            im0 = imq.get()
            print("twice")
            # if (recondone_queue.value == 1):
            temp_cnt = track_cnt.value
            track_cnt.value = temp_cnt + 1
            if (initbox.qsize() != 0):
                if (init_flag != 1):
                    init_box = initbox.get(block = False)
                    # print("reinit")
                    tracker_obj = cv2.TrackerCSRT_create()
                    tracker_obj.init(im0, tuple(
                        [init_box[0], init_box[1], init_box[2] - init_box[0], init_box[3] - init_box[1]]))
                    init_flag = 1
                is_update_ok, tbox = tracker_obj.update(im0)
                if (is_update_ok):
                    cv2.rectangle(im0, (int(tbox[0]), int(tbox[1])), (int(tbox[0] + tbox[2]), int(tbox[1] + tbox[3])),
                                  (255, 0, 0), 2, 1)
                    t1 = (int(tbox[0]), int(tbox[1]))
                    t2 = int(tbox[0] + tbox[2]), int(tbox[1] + tbox[3])
                    global_color = str(color_queue.value, encoding="utf-8")
                    # global_color = color_queue.Value
                    if global_color is not None:
                        color_check = check_color((im0[t1[1] + 2:t2[1] - 2, t1[0] + 2:t2[0] - 2]), global_color)
                    # print("color check result is %s" % color_check)
                    boxq.put((t1[0],t1[1],t2[0],t2[1]))
                    boxq.get() if boxq.qsize() > 1 else time.sleep(0.01)
                if (track_cnt.value % 100 == 0 or color_check != 1):
                    init_flag = 0
                    # recondone_queue.value = 0
                    print("tracking %s"%track_cnt)
    except Exception as e:
        print(Exception, ": in tracking ",e)


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


def Serial_threading():
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
            print("distance is %s"%global_dist)
    except:
        print("error")


def init_detect():
    img_size = 416
    webcam = '0'
    device = torch_utils.select_device(force_cpu=ONNX_EXPORT)
    torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
    model = Darknet(config.cfg, img_size)
    # Load weights
    model.load_state_dict(torch.load(config.weights, map_location=device)['model'])
    # Eval mode
    model.to(device).eval()
    half = True and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()
    # dataset = LoadWebcam(config.source, img_size=img_size, half=half)
    classes = load_classes(parse_data_cfg(config.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    return device,model,classes,colors





def run_single_camera():
    device, model, classes, colors = init_detect()
    mp.set_start_method(method='spawn')  # init
    img_queue = mp.Queue(maxsize=5)
    initbox = mp.Queue(maxsize = 3)
    boxq = mp.Queue(maxsize=3)



    dist_queue = mp.Queue(maxsize=3)
    # track_cnt_queue = mp.Queue(maxsize=3)

    recondone_queue = mp.Value("d", 0)
    track_cnt = mp.Value("d", 0)
    color_queue = mp.Value(ctypes.c_char_p,b'white')

    processes = [mp.Process(target=image_put, args=(img_queue,)),
                 mp.Process(target=image_get, args=(img_queue, boxq, 'Cam0')),
                 mp.Process(target=torch_recog_threading,
                            args=(img_queue, boxq, initbox, device, model, classes, colors,color_queue,recondone_queue,)),
                 mp.Process(target=tracking_thread, args=(initbox, img_queue, boxq,color_queue,track_cnt,recondone_queue,))
                 ]

    [process.start() for process in processes]
    [process.join() for process in processes]



def run():
    run_single_camera()  #
    pass


if __name__ == '__main__':
    run()