from utils.datasets import *
from utils.utils import *
import re
import post_process


def get_recbox(x, img,label=None):
    list_1 = [0, [0, 0], [0, 0], 0]
    list_2 = [0, [0, 0], [0, 0], 0]
    # Plots one bounding box on image img
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    list = [list_1, list_2]
    name_list = ['balloon', 'ball']
    box = img[c1[1]:c2[1], c1[0]:c2[0]]
    box_01 = img[c1[1]:int((c2[1]+c1[1])/2), c1[0]:c2[0]]
    box_02 = img[int((c2[1]+c1[1])/2):c2[1], c1[0]:c2[0]]
    box_03 = img[c1[1]:c2[1], c1[0]:int((c2[0]+c1[0])/2)]
    box_04 = img[c1[1]:c2[1], int((c2[0]+c1[0])/2):c2[0]]
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
    return list, box,box_01,box_02,box_03,box_04


def set_size(im0,half):
    img, *_ = letterbox(im0, 416)
    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float16 if half else np.float32)  # uint8 to fp16/fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return img


def check_color(box,target_color):
    try:
        _,color = post_process.get_color(box)
    except:
        return 0
    else:
        if (color == target_color):
            print("match success")
            return 1
        else:
            print("match failed")
            return 0


def set_front(str,mode):
    if (mode == 1):
        output = "\033[1;31m" + str + "\033[0m"
    elif (mode == 2):
        output = "\033[1;32m" + str + "\033[0m"
    elif (mode == 3):
        output = "\033[1;34m" + str + "\033[0m"
    return output


# limited speed
def suppress(speed,target_speed):
    if (speed>target_speed):
        speed = target_speed
    elif(speed<-target_speed):
        speed = -target_speed
    # x = speed
    # speed = 1 / (1 + np.exp(-x))
    return int(speed)


def check_red_mark(img):
    hue_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_range = np.array([156, 43, 46])
    high_range = np.array([180, 255, 255])
    th = cv2.inRange(hue_image, low_range, high_range)
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    return dilated



def gstreamer_pipeline (capture_width=3280, capture_height=2464, display_width=480, display_height=360, framerate=21, flip_method=0) :
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

def update_part(new_box):
    new_box_1 = new_box[0:int(new_box.shape[0]/2), :]
    new_box_2 = new_box[int(new_box.shape[0]/2):new_box.shape[0], :]
    new_box_3 = new_box[:,0 :int(new_box.shape[1]/2)]
    new_box_4 = new_box[:,int(new_box.shape[1]/2) :new_box.shape[1]]
    return new_box_1,new_box_2,new_box_3,new_box_4