import cv2
import numpy as np

def find_circle(frame):
    frame = cv2.blur(frame, (5,5))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 10, 120)
    cv2.imshow('canny', canny)
    circles1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,
                                600, param1=100, param2=30, minRadius=40, maxRadius=120)
    if circles1 is not None:
        circles = circles1[0, :, :]
        circles = np.uint16(np.around(circles))
        for i in circles[:]:
            cv2.circle(frame, (i[0], i[1]), i[2], (255, 0, 0), 5)
            cv2.circle(frame, (i[0], i[1]), 2, (255, 0, 255), 10)
            cv2.rectangle(frame, (i[0] - i[2], i[1] + i[2]), (i[0] + i[2], i[1] - i[2]), (255, 255, 0), 5)
            cv2.imshow('rec', frame)
        pass
    print(circles1)

"""
   
"""
def detect_circle():
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        ret, frame = cap.read()
        # cv2.flip(frame, 0, frame) # uncomment in jetson nano
        if ret:
            find_circle(frame)
            # cv2.imshow('frame', frame)
            if cv2.waitKey(1) == 27:  # esc to quit
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    detect_circle()