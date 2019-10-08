import cv2
import numpy as np
import colorList

filename = 'car04.jpg'


# 处理图片
def get_color(frame):
    print('go in get_color')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maxsum = -100
    color = None
    color_dict = colorList.getColorList()
    cntm = []
    for d in color_dict:
        mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
        cv2.imwrite(d + '.jpg', mask)
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary, None, iterations=2)
        cnts, hiera = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sum = 0
        for c in cnts:
            sum += cv2.contourArea(c)
        if sum > maxsum:
            maxsum = sum
            color = d
            cntm = cnts
        # print(color)
    return cntm,color


def match_img(image,Target,value):
    img_rgb = image
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # template = cv2.imread(Target,0)
    template = cv2.cvtColor(Target, cv2.COLOR_BGR2GRAY)
    match = 0
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    left_top = max_loc
    right_bottom = (left_top[0] + w, left_top[1] + h)
    if max_val>=value:
        match = 1
    return match,left_top,right_bottom

    #
    # threshold = value
    # loc = np.where( res >= threshold)
    # pt = (0, 0)
    # for pt in zip(*loc[::-1]):
    #     # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255),1)
    #     match += 1
    #     cv2.imshow('Detected', img_rgb)
    #     if match ==1:
    #         break


if __name__ == '__main__':
    frame = cv2.imread(filename)
    print(get_color(frame))