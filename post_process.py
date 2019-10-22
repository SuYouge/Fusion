import cv2
import numpy as np
import collections
filename = 'car04.jpg'


def getColorList():
    dict = collections.defaultdict(list)

    # # 黑色
    # lower_black = np.array([0, 0, 0])
    # upper_black = np.array([180, 255, 46])
    # color_list = []
    # color_list.append(lower_black)
    # color_list.append(upper_black)
    # dict['black'] = color_list

    # #灰色
    # lower_gray = np.array([0, 0, 46])
    # upper_gray = np.array([180, 43, 220])
    # color_list = []
    # color_list.append(lower_gray)
    # color_list.append(upper_gray)
    # dict['gray']=color_list

    # 白色
    # lower_white = np.array([0, 0, 221])
    # upper_white = np.array([180, 30, 255])
    # color_list = []
    # color_list.append(lower_white)
    # color_list.append(upper_white)
    # dict['white'] = color_list

    # 红色
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red'] = color_list

    # 红色2
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red2'] = color_list

    # 橙色
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([25, 255, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    dict['orange'] = color_list

    # 黄色
    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list

    # 绿色
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list

    # 青色
    lower_cyan = np.array([78, 43, 46])
    upper_cyan = np.array([99, 255, 255])
    color_list = []
    color_list.append(lower_cyan)
    color_list.append(upper_cyan)
    dict['cyan'] = color_list

    # 蓝色
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    dict['blue'] = color_list

    # 紫色
    lower_purple = np.array([125, 43, 46])
    upper_purple = np.array([155, 255, 255])
    color_list = []
    color_list.append(lower_purple)
    color_list.append(upper_purple)
    dict['purple'] = color_list

    return dict


# 处理图片
def get_color(frame):
    # print('go in get_color')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    maxsum = -100
    color = None
    color_dict = getColorList()
    cntm = []
    for d in color_dict:
        mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
        cv2.imwrite('cache_pic/' + d + '.jpg', mask)
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary, None, iterations=2)
        if (cv2.__version__=='4.1.1'):
            cnts, hiera = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            img, cnts, hiera = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sum = 0
        for c in cnts:
            sum += cv2.contourArea(c)
        if sum > maxsum:
            maxsum = sum
            color = d
            cntm = cnts
    print(color)
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
    new_box1 = image[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]
    new_box2 = image[left_top[1]:int((right_bottom[1]+left_top[1])/2), left_top[0]:right_bottom[0]]
    new_box3 = image[int((right_bottom[1]+left_top[1])/2):right_bottom[1], left_top[0]:right_bottom[0]]
    new_box4 = image[left_top[1]:right_bottom[1], left_top[0]:int((right_bottom[0]+left_top[0])/2)]
    new_box5 = image[left_top[1]:right_bottom[1], int((right_bottom[0]+left_top[0])/2):right_bottom[0]]
    return match,left_top,right_bottom,new_box1,new_box2,new_box3,new_box4,new_box5


def multi_scale_match(image,Target,value):
    template = Target
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # template = cv2.Canny(template, 50, 200)
    tH = template.shape[0]
    tW = template.shape[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None
    match = 0
    # for i in range(10):
    for scale in (np.linspace(0.2, 1, 10)[::-1]).tolist():
        # resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        resized = cv2.resize(gray, (0, 0), fx=scale,fy=scale, interpolation=cv2.INTER_NEAREST)
        r = gray.shape[1] / float(resized.shape[1])
        # 如果裁剪之后的图片小于模板的大小直接退出
        if ((resized.shape[0] < tH) or (resized.shape[1] < tW)):
            # print("template is too big")
            break
        # 首先进行边缘检测，然后执行模板检测，接着获取最小外接矩形
        # edged = cv2.Canny(resized, 50, 200)
        # res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        # result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        # 如果发现一个新的关联值则进行更新
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
    maxVal, maxLoc, r = found
    if maxVal>=value:
        match = 1
    left_top = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    right_bottom = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    new_box1 = image[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]
    new_box2 = image[left_top[1]:int((right_bottom[1] + left_top[1]) / 2), left_top[0]:right_bottom[0]]
    new_box3 = image[int((right_bottom[1] + left_top[1]) / 2):right_bottom[1], left_top[0]:right_bottom[0]]
    new_box4 = image[left_top[1]:right_bottom[1], left_top[0]:int((right_bottom[0] + left_top[0]) / 2)]
    new_box5 = image[left_top[1]:right_bottom[1], int((right_bottom[0] + left_top[0]) / 2):right_bottom[0]]
    print("match const is %s"%maxVal)
    return match,left_top,right_bottom,new_box1,new_box2,new_box3,new_box4,new_box5


if __name__ == '__main__':
    frame = cv2.imread(filename)
    print(get_color(frame))