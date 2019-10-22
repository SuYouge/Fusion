import cv2
import numpy as np

mode = 1
def detect_reball():
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        ret, frame = cap.read()
        # cv2.flip(frame, 0, frame)
        if ret:
            cv2.imshow('origin', frame)
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
            elif (mode == 3):
                low_range = np.array([125, 43, 46])
                high_range = np.array([155, 255, 255])
            elif (mode == 4):
                low_range = np.array([11, 43, 46])
                high_range = np.array([25, 255, 255])
            elif (mode == 5):
                low_range = np.array([35, 43, 46])
                high_range = np.array([77, 255, 255])
            th = cv2.inRange(hue_image, low_range, high_range)
            cv2.imshow('thresh', th)
            dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
            cv2.imshow('dilated', dilated )
            circles = cv2.HoughCircles(dilated, cv2.HOUGH_GRADIENT, 1, 100, param1=15, param2=7, minRadius=10,
                                       maxRadius=20)
            if circles is not None:
                x, y, radius = circles[0][0]
                center = (x, y)
                cv2.circle(frame, center, radius, (0, 255, 0), 2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_reball()