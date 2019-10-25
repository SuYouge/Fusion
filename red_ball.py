import cv2
import numpy as np


def bounding(opening):
    if (cv2.__version__).split('.')[0] == '3':
        _, contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cnt = contours[np.argmax([cv2.contourArea(cnt) for cnt in contours])]
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(area)
    for i in range(max_idx - 1):
        cv2.fillConvexPoly(opening, contours[max_idx - 1], 0)
    cv2.fillConvexPoly(opening, contours[max_idx], 255)
    cnt = contours[max_idx]
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(opening, (x, y), (x + w, y + h), (0, 255, 0), 2)

def gamma_trans(img, gamma):
    # hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
    # hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
    # hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
    # 具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # 实现映射用的是Opencv的查表函数
    img0 = cv2.LUT(img, gamma_table)
    return img0

mode = 3
def detect_reball():
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        ret, frame = cap.read()
        # cv2.flip(frame, 0, frame)
        if ret:
            frame = gamma_trans(frame, 0.5)
            cv2.imshow('origin', frame)
            # equ = cv2.equalizeHist(frame)
            # res = np.hstack((frame, equ))
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #             # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            #             # res = clahe.apply(frame)
            #             # cv2.imshow('res', res)
            hue_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            if (mode == 1):  #'red'
            # low_range = np.array([0, 123, 100])
            # high_range = np.array([5, 255, 255])
            # red
                low_range = np.array([156, 43, 46])
                high_range = np.array([180, 255, 255])
            elif (mode == 2): # 'blue'
                low_range = np.array([100, 43, 46])
                high_range = np.array([124, 255, 255])
            elif (mode == 3): #pp
                low_range = np.array([125, 43, 46])
                high_range = np.array([155, 255, 255])
            elif (mode == 4):
                low_range = np.array([11, 43, 46])
                high_range = np.array([25, 255, 255])
            elif (mode == 5):
                low_range = np.array([35, 43, 46])
                high_range = np.array([77, 255, 255])
            th = cv2.inRange(hue_image, low_range, high_range)
            # cv2.imshow('thresh', th)
            # dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
            # cv2.imshow('dilated', dilated )
            # open operation
            kernel = np.ones((8, 8), np.uint8)
            opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
            if (cv2.__version__).split('.')[0] == '3':
                _, contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # print(cv2.__version__)
            else:
                contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # print(cv2.__version__)
            # cnt = contours[np.argmax([cv2.contourArea(cnt) for cnt in contours])]
            if contours:
                # print("in contour")
                area = []
                for i in range(len(contours)):
                    area.append(cv2.contourArea(contours[i]))
                max_idx = np.argmax(area)
                for i in range(max_idx - 1):
                    cv2.fillConvexPoly(opening, contours[max_idx - 1], 0)
                cv2.fillConvexPoly(opening, contours[max_idx], 255)
                cnt = contours[max_idx]
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(opening, (x, y), (x + w, y + h), (0, 255, 0), 20)
            # else:
            #     x, y, w, h = cv2.boundingRect(opening)
            #     cv2.rectangle(opening, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow('opening', opening)
            cv2.imshow('thresh', th)
            # print(cnt)
            # cv2.drawContours(opening, contours, -1, (0, 255, 0), 3)
            # circles = cv2.HoughCircles(dilated, cv2.HOUGH_GRADIENT, 1, 100, param1=15, param2=7, minRadius=10,
            #                            maxRadius=20)
            # if circles is not None:
            #     x, y, radius = circles[0][0]
            #     center = (x, y)
            #     cv2.circle(frame, center, radius, (0, 255, 0), 2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_reball()