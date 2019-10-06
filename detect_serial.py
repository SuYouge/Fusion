import time
import serial
import struct
import threading
import random
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

img_size = 416
out = 'output'
weights = 'weights/best.pt'
# half = True
source = '0'
cfg = 'cfg/yolov3-tiny-2cls.cfg'
data = 'data/ball.data'
conf_thres = 0.8
nms_thres = 0.2

cache_box = []
cache_size = 3

global_dist = 0
global_speedx = 0
global_speedy = 0 # if speedy >0 move forward else backward
global_speedr = 0 # if speedr >0 turn left else turn right
global_decpos = [[0,0,0,0],[0,0,0,0]] # [obj1[p_upleft,p_downright], obj2[p_upleft,p_downright]]
global_imgsize = [0,0] # [width,height]

global_dist_thresh = 200

mode = ['init','init']
# mode_list = ['init','wander', 'follow', 'find', 'attack', 'rehearsal']

# /dev/ttyUSB0 for ubuntu
serialPort = "COM14"  # serial no
baudRate = 9600  # Baudrate

test_mode = True
# list = [[0, [0, 0], [0, 0], 0], \
#         [0, [0, 0], [0, 0], 0]]
'''
wander : avoid barrel and looking for target
follow : focus on target and keep distance
find   : find target
attack : approach to target 
rehearsal : undefined
'''

'''
shape of list
    list = [[0, [0, 0], [0, 0], 0] balloon
           [0, [0, 0], [0, 0], 0]] ball
list cache
'''


# limited speed
def suppress(speed):
    if (speed>1000):
        speed = 1000
    elif(speed<-1000):
        speed = -1000
    # x = speed
    # speed = 1 / (1 + np.exp(-x))
    return speed


# serial port operation
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
            print("speedx = %s, speedy = %s, speedr = %s \n"%(global_speedx,global_speedy,global_speedr))
        # return n

    def read_data(self):
        global global_dist
        while True:
            # self.message = self.port.readline()
            self.Bytedist = self.port.read(13)
            str_dist = str(self.Bytedist, encoding="utf-8")
            self.message = int(re.findall(r'\d+', str_dist)[0])
            global_dist = self.message
            # print(self.message)


# make decision
def decision(list = None):
    # calculate decision FSM value
    global global_dist
    global mode
    # print(list, global_dist)
    #  FSM
    if (mode[0] == 'init'):
        rod = random.randint(0, 9)
        if (rod<=4):
            mode[1] = 'wander'
            execute(list)
        elif (rod>4):
            mode[1] = 'find'
            execute(list)
    elif (mode[0] == 'find'):
        if ():
            mode[1] = 'attack'
            execute(list)
        elif ():
            mode[1] = 'follow'
            execute(list)
        elif ():
            mode[1] = 'wander'
            execute(list)
    elif (mode[0] == 'follow'):
        if ():
            mode[1] = 'attack'
            execute(list)
        elif ():
            mode[1] = 'find'
            execute(list)
        elif ():
            mode[1] = 'wander'
            execute(list)
    elif (mode[0] == 'wander'):
        if ():
            mode[1] = 'attack'
            execute()
        elif ():
            mode[1] = 'follow'
            execute(list)
    elif (mode[0] == 'attack'):
        if ():
            mode[1] = 'find'
        elif():
            mode[1] = 'follow'
    else:
        mode[1] = mode[0]
        execute(list)


def execute(list = None):
    # define a decision filter(list[]) for every detection
    global mode
    global global_dist
    global global_speedx
    global global_speedy
    global global_speedr
    global global_decpos
    global global_imgsize

    if (mode[1] == 'init'):
        global_speedx = 0
        global_speedy = 0
        global_speedr = 0
        # update cache target
    elif(mode[1] == 'wander'):
        rander = random.randint(0, 8)
        if (global_dist>global_dist_thresh):
            if (rander<=2):
                global_speedx = 0
                global_speedy = 500
                global_speedr = 0
            elif (rander>2 & rander <=5):
                global_speedx = 0
                global_speedy = 300
                global_speedr = 100
            else:
                global_speedx = 0
                global_speedy = 300
                global_speedr = -100
        elif (global_dist<global_dist_thresh):
            global_speedx = 0
            global_speedy = -500
            global_speedr = 0
        # update cache target
    elif (mode[1] == 'follow'):
        if (global_dist<global_dist_thresh):
            kr = 0.5
            global_speedx = 200
            global_speedy = 0
            global_speedr = suppress((global_imgsize[0] - (global_decpos[0][0] + global_decpos[0][2]) / 2) * kr)
        else:
            kr = 0.25
            global_speedx = -200
            global_speedy = 0
            global_speedr = suppress((global_imgsize[0] - (global_decpos[0][0] + global_decpos[0][2]) / 2) * kr)
        # update cache target
    elif (mode[1] == 'find'):
        if (): # balloon disappear from left
            global_speedr = 500 # define speedr based on test
        elif (): #balloon disappear from right
            global_speedr = -500
        # update cache target
    elif (mode[1] == 'attack'):
        if (global_dist>=80):
            global_speedx = 0
            global_speedy = 700
            global_speedr = 0
        # update cache target
    elif (mode[1] == 'rehearsal'):
        global_speedx = 0
        global_speedy = 0
        global_speedr = 0

    print('executed state is %s, previous state is %s\n speed=(%s,%s,%s) dist =%s\n'\
          %(mode[1],mode[0],global_speedx,global_speedy,global_speedr,global_dist))
    mode[0] = mode[1]

def mode_test():
    # print("is %s" % cache_box)
    center = cal_ave()
    print(center)
    pass


def cal_ave():
    global cache_box
    x1,y1 = 0,0
    x2,y2 = 0,0
    p = 0
    cnt = 0
    center = [(0,0),0]
    if len(cache_box):
        for i in range(len(cache_box)):
            if (cache_box[i][0][0] == 1):
                x1 = x1 + cache_box[i][0][1][0]
                y1 = y1 + cache_box[i][0][1][1]
                # print("cache box i is %s" % cache_box[i][0][1][0])
                x2 = x2 + cache_box[i][0][2][0]
                y2 = y2 + cache_box[i][0][2][1]
                # print("cache box i is %s" % cache_box[i][0][2][0])
                p = p + cache_box[i][0][3]
                # print("cache box i is %s" % cache_box[i][0][3])
                cnt = cnt + 1
                center = [((x1/cnt+x2/cnt)/2,(y1/cnt+y2/cnt)/2),p/cnt]
        else:
            pass
    else:
        pass
    return center

def inqueue(box):
    global cache_box
    if len(cache_box)<= cache_size-1:
        cache_box.append(box)
    elif len(cache_box) > cache_size-1:
        # print(">3")
        cache_box.append(box)
        for i in range(1):  ##左移
            cache_box.insert(len(cache_box), cache_box[0])
            cache_box.remove(cache_box[0])
            cache_box.pop()
    # ave_box = cal_average(cache_box)
    # print("is %s"%cache_box)

def detect(save_img=False, stream_img=False):
    img_size = 416
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http')
    # Initialize
    device = torch_utils.select_device(force_cpu=ONNX_EXPORT)
    torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    # Initialize model
    model = Darknet(cfg, img_size)
    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Eval mode
    model.to(device).eval()

    # Half precision
    half = True and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    if webcam:
        stream_img = True
        dataset = LoadWebcam(source, img_size=img_size, half=half)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)
    # Get classes and colors
    classes = load_classes(parse_data_cfg(data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    # Run inference
    t0 = time.time()
    for path, img, im0, vid_cap in dataset:
        t = time.time()
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        pred, _ = model(img)
        # non_max_suppression (x1, y1, x2, y2, object_conf, class_conf, class)
        det = non_max_suppression(pred.float(), conf_thres, nms_thres)[0]
        s = '%gx%g ' % img.shape[2:]  # print string

        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # print(det[:, :4])
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, classes[int(c)])  # add to string
            # Write results
            for *xyxy, conf, _, cls in det:
                if save_img or stream_img:  # Add bbox to image
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    # print("label is %s"%label)
                    list= plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
            # print(list)
        else:
            list = [[0, [0, 0], [0, 0], 0], \
                    [0, [0, 0], [0, 0], 0]]
        #####################TEMP####################
        inqueue(list)
        mode_test()
        #############################################
        if test_mode is not True:
            mSerial.get_dist()
            # mode_test()
        #############################################
        print('%sDone. (%.3fs)' % (s, time.time() - t))
        # Stream results
        if stream_img:
            cv2.imshow(weights, im0)
            if cv2.waitKey(1) == 27:  # esc to quit
                cv2.destroyAllWindows()
                break
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    if test_mode is not True:
        mSerial = SerialPort(serialPort, baudRate)
        t1 = threading.Thread(target=mSerial.read_data)
        t1.start()
        t2 = threading.Thread(target=mSerial.set_speed)
        t2.start()
    with torch.no_grad():
        detect()
