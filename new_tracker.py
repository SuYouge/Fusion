import time
import serial
import struct
import threading
import post_process
import config
import random

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


temp_cache = []
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


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
    dataset = LoadWebcam(config.source, img_size=img_size, half=half)
    classes = load_classes(parse_data_cfg(config.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    return dataset,device,model,classes,colors

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


def balloon_tracker():
    recodone_flag = 0
    init_flag = 0
    track_cnt = 0
    color = None
    dataset,device,model,classes,colors = init_detect()
    for path, img, im0, vid_cap in dataset: # im0 is origin pic, img is inferenced pic
        list = [[0, [0, 0], [0, 0], 0], \
                [0, [0, 0], [0, 0], 0]]
        cv2.imshow("origin", im0)
        # recognition
        if(recodone_flag == 0):
            print("in recognition")
            img = torch.from_numpy(img).unsqueeze(0).to(device)
            pred, _ = model(img)
            # non_max_suppression (x1, y1, x2, y2, object_conf, class_conf, class)
            det = non_max_suppression(pred.float(), config.conf_thres, config.nms_thres)[0]
            s = '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # add to string
                for *xyxy, conf, _, cls in det:
                        label = '%s %.2f' % (classes[int(cls)], conf)
                        # print("label is %s"%label)
                        list,box = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                        # color check
                        if (classes[int(cls)] == 'balloon'):
                            cntm, color = post_process.get_color(box)
                            if (cntm is not None):
                                print("cnt ok and color is %s" % color)
                                match = post_process.match_img(im0, box, 0.8)
                                temp_cache = box
                                recodone_flag = 1
                                print("flag set")
                                init_box = xyxy
                                cv2.imshow("box", box)
                                # print([init_box[0],init_box[1],init_box[2]-init_box[0],init_box[3]-init_box[1]])
        elif(recodone_flag == 1 ):
            track_cnt += 1
            print("tracker cnt %s"%track_cnt)
            print("in tracker")
            if (init_flag != 1):
                tracker_obj = cv2.TrackerCSRT_create()
                tracker_obj.init(im0, tuple([init_box[0],init_box[1],init_box[2]-init_box[0],init_box[3]-init_box[1]]))
                init_flag = 1
            is_update_ok, tbox = tracker_obj.update(im0)
            print(is_update_ok)
            color_check = 1
            if (is_update_ok):
                cv2.rectangle(im0, (int(tbox[0]), int(tbox[1])), (int(tbox[0] + tbox[2]), int(tbox[1] + tbox[3])), (255, 0, 0), 2, 1)
                t1 = (int(tbox[0]),int(tbox[1]))
                t2 = int(tbox[0] + tbox[2]), int(tbox[1] + tbox[3])
                if color is not None:
                    color_check = check_color((im0[t1[1]+2:t2[1]-2, t1[0]+2:t2[0]-2]),color)
                print("color check result is %s"%color_check)
            if (track_cnt%50 == 0 or color_check != 1):
                recodone_flag = 0
                init_flag = 0
        cv2.imshow(config.weights, im0)
        if cv2.waitKey(1) == 27:  # esc to quit
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    with torch.no_grad():
        balloon_tracker()