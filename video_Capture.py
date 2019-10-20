import cv2
import multiprocessing as mp
import time

def image_put(q,):
    cap = cv2.VideoCapture(0)
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


def image_get(q,):
    cv2.namedWindow('Cam0', flags=cv2.WINDOW_FREERATIO)
    try:
        while True:
            im0 = q.get()
            cv2.imshow('Cam0', im0)
            cv2.waitKey(1)
    except Exception as e:
        print(Exception, ": in image get ", e)

def run_single_camera():

    mp.set_start_method(method='spawn')  # init
    img_queue = mp.Queue()

    processes = [
                 mp.Process(target=image_put, args=(img_queue,)),
                 mp.Process(target=image_get, args=(img_queue, )),
                 ]

    [process.start() for process in processes]
    [process.join() for process in processes]

def run():
    run_single_camera()  #
    pass


if __name__ == '__main__':
    run()