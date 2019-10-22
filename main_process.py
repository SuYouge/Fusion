import time
import multiprocessing as mp
import serial
import platform
import struct
import config
import threading
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

cache_box = []
cache_size = 3
shake_flag = 0
shake_cnt = 0
diappear_flag = -1
forward_flag = 1

mode_flag = 1
round_flag =1


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
            center = ((0,0),0)
            last_step = 0
            # pcnt = 0
            while True:
                step = time.clock()
                mSerial.get_dist()
                # if (mode_flag == 2):
                #     wander_mode()
                # if(mode_flag == 1):
                #     center,last_step,speed = mode_test(step,center,last_step)
                _ = wander_test()
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


# def wander_mode():
#     balloon_center = cal_ave_balloon()
#     foam_center = cal_ave_foam()
#     if (balloon_center[0] == (0, 0)):
#         if(global_dist>500) and (forward_flag == 1):
#             set_speed(0, 200, 0)
#         elif (round_flag == 1):
#             set_speed(0, 200, 0)
#             round_cnt+=1
#     else:
#         mode_flag = 1
#     pass


def counter(target,cnt):
    if (cnt<target):
        cnt = cnt + 1
        return cnt,False
    else:
        return 0,True


def match_pattern_list(im0,temp_cache_1, temp_cache_2, temp_cache_3, temp_cache_4):
    match2, c1, c2, new_box, _, _, _, _ = post_process.match_img(im0, temp_cache_2, 0.75)
    if (match2==1):
        ret = 1
        cache = temp_cache_2
        return ret, cache,'down'
    match1, c1, c2, new_box, _, _, _, _ = post_process.match_img(im0, temp_cache_1, 0.75)
    if (match1==1):
        ret = 1
        cache = temp_cache_1
        return ret, cache,'up'
    match3, c1, c2, new_box, _, _, _, _ = post_process.match_img(im0, temp_cache_3, 0.75)
    if (match3==1):
        ret = 1
        cache = temp_cache_3
        return ret, cache,'left'
    match4, c1, c2, new_box, _, _, _, _ = post_process.match_img(im0, temp_cache_4, 0.75)
    if (match4==1):
        ret = 1
        cache = temp_cache_4
        return ret, cache,'right'
    ret = 0
    cache = []
    return ret, cache,'none'



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


def shake(times):
    global shake_cnt
    global shake_flag
    if (times != 0):
        if (shake_cnt <= times):
            set_speed(0, 0, 500)
            shake_cnt += 1
        elif(times<=shake_cnt <= 2*times):
            set_speed(0, 0, -500)
            shake_cnt += 1
        else:
            shake_cnt = 0
            shake_flag = 0
        print(set_front("shaking",3))
    pass


def wander_test():
    # Finding mode
    balloon_center, balloon_size = cal_ave_balloon()
    print(set_front(("balloon size is  %s" % balloon_size), 2))
    foam_center = cal_ave_foam()
    speed = 0
    if (balloon_center[0] == (0, 0)):
        speed = set_speed(0, 0, diappear_flag * 200)
        print("finding in %s\n" % diappear_flag)
        # if target was found , but lost check found_flag
    else:
        if (balloon_center[0][0] < 0.35 or balloon_center[0][0] > 0.65) and (balloon_size>0.3):
            centering_speed = (balloon_center[0][0] - 0.5) * 400
            print("low speed centering speed %s" % centering_speed)
            speed = set_speed(0, 0, suppress(centering_speed, 350))
            print("centering")
        elif (balloon_center[0][0] < 0.35 or balloon_center[0][0] > 0.65) and (balloon_size<=0.3):
            centering_speed = (balloon_center[0][0] - 0.5) * 200
            print("high speed centering speed %s" % centering_speed)
            speed = set_speed(0, 0, suppress(centering_speed, 150))
            print("centering")
        elif (0.35 <= balloon_center[0][0] <= 0.65):
            # set_speed(0, 0, 0)
            print("incenter")
            if ((global_dist > 400) and (balloon_size<0.3)):
                centering_speed = (balloon_center[0][0] - 0.5) * 200
                speed = set_speed(0, 500, suppress(centering_speed, 200))
                # print("approaching %s"%global_dist)
                print(set_front(("approaching %s" % global_dist), 1))
                # check
            elif ((global_dist > 400) and (balloon_size>=0.3)):
                centering_speed = (balloon_center[0][0] - 0.5) * 200
                speed = set_speed(0, 300, suppress(centering_speed, 200))
            else:
                print(set_front("Found", 1))
                speed = set_speed(0, 0, 0)
    return speed


# Finding mode
def mode_test(step,last_center,last_step):
    global shake_flag
    # Finding mode
    balloon_center,balloon_size = cal_ave_balloon()
    foam_center = cal_ave_foam()
    print('%s is step'%(step - last_step))
    # if ((step - last_step)>0):
    #     speed = ((center[0][0]-last_center[0][0])/(step - last_step))*10e3
    # else:
    #     speed = 0
    # print(set_front("speed between two frame is %s" % speed ,1))
    if ((balloon_center[0] == (0,0)) and (shake_flag != 1)):
        speed = set_speed(0, 0, diappear_flag*300)
        print("finding in %s\n"%diappear_flag)
        # if target was found , but lost check found_flag
    elif(shake_flag == 1):
        shake(1)
        shake_flag = 0
    else:
        if (balloon_center[0][0]<0.35 or balloon_center[0][0]>0.65):
            centering_speed = (balloon_center[0][0]-0.5)*300
            print("centering speed %s"%centering_speed)
            speed = set_speed(0, 0, suppress(centering_speed, 200))
            print("centering")
        elif(0.35<=balloon_center[0][0]<=0.65):
            # set_speed(0, 0, 0)
            print("incenter")
            if(global_dist>500):
                centering_speed = (balloon_center[0][0] - 0.5) * 50
                speed = set_speed(0, 200, suppress(centering_speed, 50))
                # print("approaching %s"%global_dist)
                print(set_front(("approaching %s" % global_dist),1))
                # check
            else:
                shake_flag = 1
        print(set_front("Found",1))
    last_step = step
    print(balloon_center)
    return balloon_center,last_step,speed


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
    cap = cv2.VideoCapture(0) if platform.system() == 'Windows' else cv2.VideoCapture(gstreamer_pipeline(flip_method=0),cv2.CAP_GSTREAMER)
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
    det_cnt = 0
    in_part_1 = 0
    match_mode = 'none'
    try:
        while True:
            im0 = q.get()
            img = set_size(im0, half)
            det_cnt += 1
            list = [[0, [0, 0], [0, 0], 0], \
                    [0, [0, 0], [0, 0], 0]]
            if (len(temp_cache) == 0 or det_cnt % 20 == 0):
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
                        list, box, temp_cache_1, temp_cache_2, temp_cache_3, temp_cache_4 = get_recbox(xyxy, im0, label=label)
                        # list_queue.put(list)
                        # cache.put(list) # if get immediately no more list in cache
                        if (classes[int(cls)] == 'balloon'):
                            cntm, color = post_process.get_color(box)
                            target_color = color
                            print("target color is %s"%target_color)
                            if (cntm is not None):
                                match = post_process.match_img(im0, box, 0.8)
                                temp_cache = box
                                in_part = 0
                                cv2.rectangle(im0, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (255, 0, 0), 1)
                                match_mode = 'ALL'
                        elif (classes[int(cls)] == 'ball'):
                            cv2.rectangle(im0, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 1)
            elif (len(temp_cache) > 0):
                match, c1, c2, new_box, _, _, _, _ = post_process.match_img(im0, temp_cache, 0.70)
                if (match!=1):
                    match, c1, c2, new_box, _, _, _, _ = post_process.multi_scale_match(im0, temp_cache, 0.70)
                img = check_red_mark(im0)
                cv2.imshow('red mark', img)
                c1n, c2n = ((floatn(c1[0] / im0.shape[1]), floatn(c1[1] / im0.shape[0])),
                            (floatn(c2[0] / im0.shape[1]), floatn(c2[1] / im0.shape[0])))
                if ((match != 0) and (check_color(new_box, target_color))):
                    print(set_front("matching", 1))
                    list = [[1, [c1n[0], c1n[1]], [c2n[0], c2n[1]], 0.8], \
                            [0, [0, 0], [0, 0], 0]]
                    # list_queue.put(list)
                    cv2.rectangle(im0, (c1[0], c1[1]), (c2[0], c2[1]), (0, 0, 255), 1)
                    temp_cache = new_box
                    if (in_part_1 != 1):
                        match_mode = 'ALL'
                        temp_cache_1, temp_cache_2, temp_cache_3, temp_cache_4 = update_part(new_box)
                else:
                    ret,cache,match_mode = match_pattern_list(im0,temp_cache_1, temp_cache_2, temp_cache_3, temp_cache_4)
                    if (ret == 1):
                        temp_cache = cache
                        in_part_1 = 1
                    else:
                        match_mode = 'Lost'
            list_queue.put(list)
            cv2.putText(im0, match_mode, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
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