import cv2
import time
import numpy as np
import copy
import random

box_num = 2

def find_box(frame):
    # read img and copy

    img = copy.deepcopy(frame)
    ##cv2.imshow('img',img)

    # make img into gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ##cv2.imshow('gray',gray)

    # threshold
    ret, thre = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # cv2.imshow('thre',thre)

    # erode
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(thre, kernel)
    # erosion = cv2.erode(erosion, kernel)

    # cv2.imshow('erosion',erosion)

    # findContours
    contours, hier = cv2.findContours(erosion,
                                            cv2.RETR_LIST,
                                            # cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(erosion, contours, -1, (0, 0, 255), 3)
    cv2.imshow('erosion',erosion)
    return contours, gray


def find_box0(frame):
    box = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gradX = cv2.Sobel(gray, cv2.CV_32F, 1, 0, -1)
    gradY = cv2.Sobel(gray, cv2.CV_32F, 0, 1, -1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 160, 160, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    cnts, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts)>box_num:
        for index in range(box_num):
            c = sorted(cnts, key=cv2.contourArea, reverse=True)[index]
            rect = cv2.minAreaRect(c)
            box.append(np.int0(cv2.boxPoints(rect)))
    elif len(cnts)<=box_num:
        for index in range(len(cnts)):
            c = sorted(cnts, key=cv2.contourArea, reverse=True)[index]
            rect = cv2.minAreaRect(c)
            box.append(np.int0(cv2.boxPoints(rect)))
    return box

def gstreamer_pipeline(capture_width=3280, capture_height=2464, display_width=480, display_height=360, framerate=21,
                       flip_method=0):
    return ('nvarguscamerasrc ! '
            'video/x-raw(memory:NVMM), '
            'width=(int)%d, height=(int)%d, '
            'format=(string)NV12, framerate=(fraction)%d/1 ! '
            'nvvidconv flip-method=%d ! '
            'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
            'videoconvert ! '
            'video/x-raw, format=(string)BGR ! appsink' % (
            capture_width, capture_height, framerate, flip_method, display_width, display_height))


def detect_qrcode():
    # cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture(0)
    # fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    # now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    # fname = "data_video/" + now + r".mp4"
    # out = cv2.VideoWriter(fname, fourcc, 20.0, (480, 360))
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(3)]
    while (cap.isOpened()):
        ret, frame = cap.read()
        # cv2.flip(frame, 0, frame) # uncomment in jetson nano
        if ret:
            box = find_box0(frame)
            if len(box)>box_num:
                for index in range (box_num):
                    print("box %s is at\n"%index)
                    print(box[index])
                    cv2.drawContours(frame, [box[index]], 0, colors[index], 3)
            elif len(box)<=box_num:
                for index in range (len(box)):
                    print("box %s is at\n"%index)
                    print(box[index])
                    cv2.drawContours(frame, [box[index]], 0, colors[index], 3)
            # out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_qrcode()
