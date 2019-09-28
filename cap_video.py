import cv2
import time

def gstreamer_pipeline (capture_width=3280, capture_height=2464, display_width=480, display_height=360, framerate=21, flip_method=0) :   
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

 
 
def make_video():
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
    fname = "data_video/"+now + r".mp4"
    out = cv2.VideoWriter(fname, fourcc, 20.0, (480,360))
    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.flip(frame,0,frame)
        if ret:
            out.write(frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    make_video()
