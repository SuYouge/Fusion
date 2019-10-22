# coding=utf-8
# 导入python包
import numpy as np
import argparse
# import imutils
import glob
import cv2

# 构建并解析参数
# ap = argparse.ArgumentParser()
# ap.add_argument("-t", "--template", required=True, help="Path to template image")
# ap.add_argument("-i", "--images", required=True, help="Path to images where template will be matched")
# ap.add_argument("-v", "--visualize", help="Flag indicating whether or not to visualize each iteration")
# args = vars(ap.parse_args())
#
# # 读取模板图片
# template = cv2.imread(args["template"])
# # 转换为灰度图片
# template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# # 执行边缘检测
# template = cv2.Canny(template, 50, 200)
# (tH, tW) = template.shape[:2]
# # 显示模板
# cv2.imshow("Template", template)

# 遍历所有的图片寻找模板
def multi_scale_match(image,Target,value):
    template = Target
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template, 50, 200)
    tH = template.shape[0]
    tW = template.shape[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None
    match = 0
    # for i in range(10):
    for scale in (np.linspace(0.02, 0.1, 5)[::-1]).tolist():
        # resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        resized = cv2.resize(gray, (0, 0), fx=scale, interpolation=cv2.INTER_NEAREST)
        r = gray.shape[1] / float(resized.shape[1])
        # 如果裁剪之后的图片小于模板的大小直接退出
        if ((resized.shape[0] < tH) or (resized.shape[1] < tW)):
            print("template is too big")
            break
        # 首先进行边缘检测，然后执行模板检测，接着获取最小外接矩形
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
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
# for scale in np.linspace(0.2, 1.0, 20)[::-1]:




	# 绘制并显示结果
	# cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2
