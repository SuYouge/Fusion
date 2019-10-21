import cv2
import time

def make_video():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # 设置分辨率
    cap.set(4, 480)
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    fname = "data_video/" + now + r".avi"
    out = cv2.VideoWriter(fname, fourcc, 20.0, (640, 480))
    while (cap.isOpened()):
        ret, frame = cap.read()
        # cv2.flip(frame, 0, frame)
        if ret:
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    make_video()
