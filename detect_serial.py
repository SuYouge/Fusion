import time
import serial
import struct
import threading
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

img_size = 416
out = 'output'
weights = 'weights/tiny-thresh06.pt'
# half = True
source = '0'
cfg = 'cfg/yolov3-tiny-2cls.cfg'
data = 'data/ball.data'
conf_thres = 0.7
nms_thres = 0.1

global_dist = 0
global_speedx = 0
global_speedy = 0
global_speedr = 0

mode = ['init','init']
# mode_list = ['init','wander', 'follow', 'find', 'attack', 'rehearsal']

'''
wander : avoid barrel and looking for target
follow : focus on target and keep distance
find   : find target
attack : approach to target 
rehearsal : undefined
'''

serialPort = "COM14"  # 串口
baudRate = 9600  # 波特率


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
            print(global_speedx,global_speedy,global_speedr)
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
    print(list,global_dist)
    # calculate decision FSM value
    global mode
    #  FSM
    if (mode[0] == 'init'):
        if ():
            mode[1] = 'wander'
            execute()
        elif ():
            mode[1] = 'follow'
            execute()
        elif ():
            mode[1] = 'find'
            execute()
        elif ():
            mode[1] = 'attack'
            execute()
        elif ():
            mode[1] = 'rehearsal'
            execute()
    else:
        mode[1] = mode[0]
        execute()

def execute():
    # define a decision filter(list[]) for every detection
    global mode
    global global_dist
    global global_speedx
    global global_speedy
    global global_speedr
    if (mode[1] == 'init'):
        global_speedx = 0
        global_speedy = 0
        global_speedr = 0
    elif(mode[1] == 'wander'):
        global_speedx = 700
        global_speedy = 0
        global_speedr = 0
    elif (mode[1] == 'follow'):
        global_speedx = 700
        global_speedy = 0
        global_speedr = 0
    elif (mode[1] == 'find'):
        global_speedx = 700
        global_speedy = 0
        global_speedr = 0
    elif (mode[1] == 'attack'):
        global_speedx = 700
        global_speedy = 0
        global_speedr = 0
    elif (mode[1] == 'rehearsal'):
        global_speedx = 0
        global_speedy = 0
        global_speedr = 0
    print('executed state is %s, previous state is %s'%(mode[1],mode[0]))
    mode[0] = mode[1]


def detect(save_img=False, stream_img=False):
    global global_speedx
    global global_speedy
    global global_speedr
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
        # Get detections
        #############################################
        mSerial.get_dist()
        # global_speedx = 700
        # global_speedy = 0
        # global_speedr = 0
        #############################################
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        pred, _ = model(img)
        # non_max_suppression (x1, y1, x2, y2, object_conf, class_conf, class)
        det = non_max_suppression(pred.float(), conf_thres, nms_thres)[0]
        s = '%gx%g ' % img.shape[2:]  # print string
        list=[[0, [0, 0], [0, 0]],[0, [0, 0], [0, 0]]]
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # print(det[:, :4])
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, classes[int(c)])  # add to string
            # Write results
            list_1 = [0, [0, 0], [0, 0]]
            list_2 = [0, [0, 0], [0, 0]]
            for *xyxy, conf, _, cls in det:
                if save_img or stream_img:  # Add bbox to image
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    list= plot_one_box(xyxy, im0, label=label, color=colors[int(cls)],list_1 = list_1,list_2 = list_2)
            # print(list)
        decision(list)
        print('%sDone. (%.3fs)' % (s, time.time() - t))
        # Stream results
        if stream_img:
            cv2.imshow(weights, im0)
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    mSerial = SerialPort(serialPort, baudRate)
    t1 = threading.Thread(target=mSerial.read_data)
    t1.start()
    t2 = threading.Thread(target=mSerial.set_speed)
    t2.start()
    with torch.no_grad():
        detect()
