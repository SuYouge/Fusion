import time
import multiprocessing as mp
import serial
import platform
import struct
import config
import threading
import random
import ctypes
import re
import cv2

from models import *  # set ONNX_EXPORT in models.py
# from utils.datasets import *
# from utils.utils import *
from util_add import *

global_dist = 0
global_speedx = 0
global_speedy = 0 # if speedy >0 move forward else backward
global_speedr = 0 # if speedr >0 turn left else turn right
s_cnt = 0
d_cnt = 0
debute_flag = 0
cache_box = []
cache_size = 1
shake_flag = 0
shake_cnt = 0
diappear_flag = -1
forward_flag = 1

# mode_flag = 1
round_flag =1
w_cnt = 0
lost_cnt = 0
q_cnt = 0
glance_flag = 0
wind_cnt = 0
b_cnt = 0

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
            time.sleep(0.1)
            # self.flushOutput()
            # time.sleep(0.1)
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
            # print('这是进程{0},线程{1}'.format(mp.current_process(), threading.current_thread()))
            # print('Flag')


def serial_threading(serial_flag,list_queue,):
    while True:
        flag = serial_flag.get()
        print("Flag setting")
        time.sleep(1)
        # speed_queue.put((0, 0, 0))
        if (flag ==1): break
    if (flag == 1):
        print("Enter SerialPort Mode")
        try:
            mSerial = SerialPort(config.serialPort, config.baudRate)
            t1 = threading.Thread(target=mSerial.read_data)
            t2 = threading.Thread(target=mSerial.set_speed)
        except Exception as e:
            print(Exception, ": in Serial threading ", e)
        try:
            t1.start()
            t2.start()
            mode_flag = 1
            center = ((0,0),0)
            last_step = 0
            debute_flag = 0
            d_cnt = 0
            # pcnt = 0
            while True:
                step = time.clock()
                mSerial.get_dist()
                # if (debute_flag!=1):
                print(mode_flag)
                if (mode_flag==1):
                    mode_flag =debute()
                if (mode_flag==2):
                    speed, mode_flag = wander_test()
                if (mode_flag==3):
                    mode_flag = shake()
                if (mode_flag==4):
                    mode_flag = avoid()
                if (mode_flag==5):
                    mode_flag = quick_return()
                if (mode_flag==6):
                    mode_flag = wind()
                if (list_queue.qsize()>0):
                    list = list_queue.get()
                    # print(list)
                    inqueue(list)
                try:
                    print("distance is %s"%global_dist)
                except:
                    pass
        except Exception as e:
                print(Exception, ": in start get dist ", e)
    else:
        time.sleep(5)


def set_speed(x=0,y=0,r=0):
    global global_speedx
    global global_speedy
    global global_speedr
    global_speedx = x
    global_speedy = y
    if config.reverse is True:
        global_speedr = -r
    else:
        global_speedr = r
    speed = (x,y,r)
    return speed


def wind():
    global global_dist
    global wind_cnt
    wind_cnt += 1
    balloon_center, balloon_size = cal_ave_balloon()
    if(global_dist>=250):
        speed = set_speed(0, 500, 0) if wind_cnt%10!=0 else set_speed(0, 0, 200)
        mode_flag = 6
    elif((balloon_center[0] != (0, 0))):
        mode_flag = 2
    else:
        mode_flag = 4
    return mode_flag


def quick_return():
    global global_dist
    global q_cnt
    mode_flag = 5
    if ((global_dist>250) or (global_dist==0)):
        q_cnt += 1
        if (q_cnt <= 80):
            speed = set_speed(0, 500, 0)
            mode_flag = 5
        elif((q_cnt>80)and(q_cnt<=160)):
            speed = set_speed(0,0, -350)
            mode_flag = 5
        else:
            mode_flag = 2
    else:
        mode_flag = 4
    return mode_flag

def avoid():
    global global_dist
    mode_flag = 4
    if (global_dist<300):
        speed = set_speed(0, -200, 0)
        mode_flag = 4
    else:
        mode_flag = 2
    return mode_flag


def debute():
    global global_dist
    global d_cnt
    # global mode_flag
    mode_flag = 1
    balloon_center, balloon_size = cal_ave_balloon()
    print(set_front(("debute balloon size is  %s" % balloon_size), 3))
    print("debute counter is %s"%d_cnt)
    foam_center = cal_ave_foam()
    speed = (0, 0, 0)
    if ((global_dist>250) or (global_dist==0)):
        speed = set_speed(0, 500, 0)
        d_cnt += 1
        if (d_cnt >= 180):
            mode_flag = 2
    else:
        mode_flag = 2
    return mode_flag


def wander_test():
    global global_dist
    mode_flag = 2
    global w_cnt
    global lost_cnt
    global glance_flag
    # Finding mode
    global b_cnt
    balloon_center, balloon_size = cal_ave_balloon()
    print(set_front(("balloon size is  %s" % balloon_size), 2))
    foam_center = cal_ave_foam()
    speed = (0,0,0)
    # if (w_cnt%10!=0):
    if (global_dist>200 and global_dist <1400):
        if (balloon_center[0] == (0, 0)):
            lost_cnt+=1
            if(lost_cnt<=60):
                    speed = set_speed(0, 0, diappear_flag * config.roundspeed)
                    print("finding in %s\n" % diappear_flag)
            else:
                if(global_dist<250):
                    speed = set_speed(0, -200, 0)
                elif(global_dist>400):
                    lost_cnt = 20
                    speed = set_speed(0, 500, 0)
                else:
                    speed = set_speed(0, 0, diappear_flag * 250)
            # if target was found , but lost check found_flag
        elif ((balloon_center[0] != (0, 0))and(global_dist>350)and(global_dist<1400)):
            glance_flag +=1
            lost_cnt = 0
            centering_speed = (balloon_center[0][0] - 0.5) * 200
            speed = set_speed(0, 700, 0)
            # speed = set_speed(0, 500, suppress(centering_speed, 80))
            if (glance_flag%5==0):
                mode_flag = 5
        elif((balloon_center[0] != (0, 0))and(global_dist<350)):
            glance_flag += 1
            lost_cnt = 0
            mode_flag = 3
            if (glance_flag%5==0):
                mode_flag = 5
    else:
        if(global_dist>1400):
            b_cnt += 1
            if (b_cnt<30):
                speed = set_speed(0, -300, 0)
                mode_flag = 2
            else:
                mode_flag = 6
        else:
            mode_flag = 4
        print("Wandering")
    return speed,mode_flag


def shake():
    global global_dist
    global s_cnt
    # global mode_flag
    mode_flag = 3
    balloon_center, balloon_size = cal_ave_balloon()
    print(set_front(("shake balloon size is  %s" % balloon_size), 1))
    print("shake counter is %s" % s_cnt)
    foam_center = cal_ave_foam()
    speed = (0, 0, 0)
    total = 20
    if (200<= global_dist < 250):
        s_cnt += 1
        if (s_cnt >= total/2):
            speed = set_speed(0, 200, 300)
            mode_flag = 3
        elif(total/2<s_cnt<=total):
            speed = set_speed(0, 200, -300)
            mode_flag = 3
        else:
            mode_flag = 2
    else:
        mode_flag = 2
    return mode_flag


def cal_ave_balloon():
    global cache_box
    global diappear_flag
    # print("%s is in box" % cache_box)
    x1,y1 = 0,0
    x2,y2 = 0,0
    p = 0
    cnt = 0
    center = [(0,0),0]
    size = 0
    if len(cache_box):
        for i in range(len(cache_box)):
            if (cache_box[i][0][0] == 1):
                x1 = x1 + cache_box[i][0][1][0]
                y1 = y1 + cache_box[i][0][1][1]
                x2 = x2 + cache_box[i][0][2][0]
                y2 = y2 + cache_box[i][0][2][1]
                p = p + cache_box[i][0][3]
                size = size + (x2 - x1)
                cnt = cnt + 1
                center = [((x1/cnt+x2/cnt)/2,(y1/cnt+y2/cnt)/2),p/cnt]
                size = size/cnt
                if (center[0][0]<=0.5):
                    diappear_flag = -1
                elif(center[0][0]>0.5):
                    diappear_flag = 1
        else:
            pass
    else:
        pass
    print(center,size)
    return center,size


def cal_ave_foam():
    global cache_box
    x1,y1 = 0,0
    x2,y2 = 0,0
    p = 0
    cnt = 0
    foam_center = [(0,0),0]
    if len(cache_box):
        for i in range(len(cache_box)):
            if (cache_box[i][1][0] == 1):
                x1 = x1 + cache_box[i][1][1][0]
                y1 = y1 + cache_box[i][1][1][1]
                x2 = x2 + cache_box[i][1][2][0]
                y2 = y2 + cache_box[i][1][2][1]
                p = p + cache_box[i][1][3]
                cnt = cnt + 1
                foam_center = [((x1/cnt+x2/cnt)/2,(y1/cnt+y2/cnt)/2),p/cnt]
        else:
            pass
    else:
        pass
    return foam_center


def inqueue(box):
    global cache_box
    if len(cache_box)<= cache_size-1:
        cache_box.append(box)
    elif len(cache_box) > cache_size-1:
        # print(">3")
        cache_box.append(box)
        for i in range(1):  #左移
            cache_box.insert(len(cache_box), cache_box[0])
            cache_box.remove(cache_box[0])
            cache_box.pop()
    # ave_box = cal_average(cache_box)
    # print("%s is in box"%cache_box)


# necessary for USB camera
def gstreamer_pipeline (capture_width=3280, capture_height=2464, display_width=480, display_height=360, framerate=21, flip_method=0) :
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))


def image_put(q,):
    if (config.camera_mode == 'USB'):
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(0) if platform.system() == 'Windows' else cv2.VideoCapture(
            gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
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


def image_get(q,list_queue,serial_flag,):
    time.sleep(3)
    device, model, classes, colors = init_detect()
    cv2.namedWindow('Cam0', flags=cv2.WINDOW_FREERATIO)
    half = True and device.type != 'cpu'
    temp_cache = []
    temp_cache_2 = []
    const_cache = []
    const_cache_2 = []
    det_cnt = 0
    in_part_1 = 0
    match_mode = 'none'
    target_color = 'blue'
    try:
        while True:
            im0 = q.get()
            im0 = gamma_trans(im0, config.gamma)
            img = set_size(im0, half)
            # img = gamma_trans(img, 0.5)
            det_cnt += 1
            list = [[0, [0, 0], [0, 0], 0], \
                    [0, [0, 0], [0, 0], 0]]
            if (len(temp_cache) == 0 or det_cnt % config.det_inter == 0 or match_mode == 'Lost'):
                print("in recognition %s"%det_cnt)
                img = torch.from_numpy(img).unsqueeze(0).to(device)
                pred, _ = model(img)
                det = non_max_suppression(pred.float(), config.conf_thres, config.nms_thres)[0]
                if (config.test_mode is not True):
                    serial_flag.put(1)
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
                        # box = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                        # list, box, temp_cache_1, temp_cache_2, temp_cache_3, temp_cache_4 = get_recbox(xyxy, im0, label=label)
                        # mark = safe_check(im0, xyxy)
                        # list_queue.put(list)
                        # cache.put(list) # if get immediately no more list in cache
                        if ((classes[int(cls)] == 'balloon')):
                            if(safe_check(im0, xyxy)):
                                list, box, box2,temp_cache_1, temp_cache_2, temp_cache_3, temp_cache_4 = get_recbox(xyxy,
                                                                                                               im0,
                                                                                                               label=label)
                                cntm, color = post_process.get_color(box)
                                target_color = color
                                print("target color is %s"%target_color)
                                if (cntm is not None):
                                    match = post_process.match_img(im0, box, 0.8)
                                    temp_cache = box
                                    const_cache = box
                                    const_cache_2 = box2
                                    in_part = 0
                                    cv2.rectangle(im0, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 255, 255), 1)
                                    match_mode = 'ALL'
                        # elif (classes[int(cls)] == 'ball'):
                        #     list, box, _, _, _, _ = get_recbox(xyxy, im0, label=label)
                        #     cv2.rectangle(im0, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 1)
            elif ((len(temp_cache) > 0)and(const_cache != [])):
                # if (const_cache != []):
                print("in const cache check")
                cv2.imshow("const",const_cache)
                match_c, c1, c2, new_box, _, _, _, _ = post_process.match_img(im0, const_cache, 0.65)
                if (match_c!=1):
                    print("const match failed")
                    match, c1, c2, new_box, _, _, _, _ = post_process.match_img(im0, temp_cache, 0.70)
                #if (match!=1):
                #    match, c1, c2, new_box, _, _, _, _ = post_process.multi_scale_match(im0, temp_cache, 0.70)
                # img = check_red_mark(im0)
                # cv2.imshow('red mark', img)
                mark = safe_check(im0, (c1[0],int(c1[1]-(c1[0]+c1[1])/2),c2[0],int(c2[1]+(c1[0]+c1[1])/2)))
                if ((match != 0) and (check_color(new_box, target_color)) and mark):
                    c1n, c2n = ((floatn(c1[0] / im0.shape[1]), floatn(c1[1] / im0.shape[0])),
                                (floatn(c2[0] / im0.shape[1]), floatn(c2[1] / im0.shape[0])))
                    print(set_front("matching", 1))
                    list = [[1, [c1n[0], c1n[1]], [c2n[0], c2n[1]], 0.8], \
                            [0, [0, 0], [0, 0], 0]]
                    # list_queue.put(list)
                    cv2.rectangle(im0, (c1[0], c1[1]), (c2[0], c2[1]), (255, 255, 255), 1)
                    temp_cache = new_box
                    if (in_part_1 != 1):
                        match_mode = 'ALL'
                        temp_cache_1, temp_cache_2, temp_cache_3, temp_cache_4 = update_part(new_box)
                else:
                    ret,cache,match_mode = match_pattern_list(im0,temp_cache_1, temp_cache_2, temp_cache_3, temp_cache_4,const_cache)
                    if (ret == 1):
                        temp_cache = cache
                        in_part_1 = 1
                    elif (config.enhance==True and (const_cache_2 != [])):
                        match_2, c12, c22, _, _, _, _, _ = post_process.match_img(im0, const_cache_2, 0.85)
                        # new_box = const_cache_2
                        if (match_2 == 1 and (check_color(const_cache_2, 'red')or check_color(const_cache_2, 'blue'))):
                            mark_2 = safe_check(im0, (c12[0], c12[1], c22[0], c22[1]),60)
                            if (mark_2 == 1):
                                c1n, c2n = ((floatn(c12[0] / im0.shape[1]), floatn(c12[1] / im0.shape[0])),
                                            (floatn(c22[0] / im0.shape[1]), floatn(c22[1] / im0.shape[0])))
                                list = [[1, [c1n[0], c1n[1]], [c2n[0], c2n[1]], 0.8], \
                                        [0, [0, 0], [0, 0], 0]]
                                # list_queue.put(list)
                                cv2.rectangle(im0, (c12[0], c12[1]), (c22[0], c22[1]), (255, 255, 255), 1)
                        else:
                            print("enhance failed")
                    else:
                        match_mode = 'Lost'
            list_queue.put(list)
            list_queue.get(list) if list_queue.qsize() > 1 else time.sleep(0.01)
            # cv2.putText(im0, match_mode, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            ############################
            if (config.camera_mode == 'USB'):
                im0 = cv2.resize(im0,(480,360))
            ############################
            cv2.imshow('Cam0', im0)
            cv2.waitKey(1)
    except Exception as e:
        print(Exception, ": in image get ", e)


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
    # device, model, classes, colors = init_detect()
    mp.set_start_method(method='spawn')  # init
    img_queue = mp.Queue()
    list_queue = mp.Queue()
    serial_flag = mp.Queue()

    processes = [
                 mp.Process(target=image_put, args=(img_queue,)),
                 mp.Process(target=image_get, args=(img_queue, list_queue,serial_flag,)),
                 mp.Process(target=serial_threading,args=(serial_flag,list_queue,)),
                 ]
    [process.start() for process in processes]
    [process.join() for process in processes]


def run():
    run_single_camera()  #
    pass


if __name__ == '__main__':
    run()
