import platform
import cv2


out = 'output'
weights = 'weights/tiny-1500-3000-best.pt'
# weights = 'weights/best.pt'
# weights = 'weights/tiny-thresh06.pt'
# half = True
source = '0'
cfg = 'cfg/yolov3-tiny-2cls.cfg'
data = 'data/ball.data'

conf_thres = 0.85
nms_thres = 0
# serialPort = "/dev/ttyUSB0"
serialPort = "COM14"  if platform.system() == 'Windows' else "/dev/ttyUSB0"# serial no /dev/ttyUSB0
# videosource =

baudRate = 9600  # Baudrate
test_mode = False
gamma = 0.5
reverse = True

camera_mode = 'USB'
# camera_mode = 'GS'


