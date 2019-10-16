import cv2
import numpy as np

# set red thresh
# lower_blue=np.array([156,43,46])
# upper_blue=np.array([180,255,255])

lower_blue = np.array([10, 43, 46])
upper_blue = np.array([100, 255, 255])

lower_cyan = np.array([78, 43, 46])
upper_cyan = np.array([255, 255, 255])

img = cv2.imread('template.jpg')

img = cv2.flip(img, 1)

# get a frame and show

frame = img

frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
cv2.imshow('Capture', frame)

# change to hsv model
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# get mask
# mask = cv2.inRange(hsv, lower_blue, upper_blue)
mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
cv2.imshow('Mask', mask)

# detect red
res = cv2.bitwise_and(frame, frame, mask=mask)
cv2.imshow('Result', res)

cv2.waitKey(0)
cv2.destroyAllWindows()