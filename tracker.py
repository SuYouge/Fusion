import cv2
import sys
interval = 1
cnt = 0
cnt_limit = 500

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[7]  # 7和4的方法比较好

if __name__ == '__main__':
    video = cv2.VideoCapture(0)

    # 打开错误时退出
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    frame_copy = frame.copy()

# 建立跟踪器，选择跟踪器的类型
    # 创建目标跟踪类的形式  tracker = cv2.Tracker + tracker_type + _create()
    # 为了同时跟踪多个目标，所以通过列表使用多个目标跟踪类
    frame_copy = frame.copy()  # 创建原图像的复制，用来显示已框选好的结果
    is_continue = 'y'  # 判断是否继续标记的符号
    tracker = []  # 目标跟踪类的列表
    bbox = []  # 图框的列表
    is_update_ok = []  # 跟踪器更新是否成功，后续使用
    n_obj = 0  # 跟踪目标的个数
    while(is_continue!='n') & (is_continue!='N'):
        tracker.append(cv2.TrackerCSRT_create())
        bbox.append(cv2.selectROI(frame_copy, False))  # bbox是矩形左上点坐标+矩形宽高，即(x,y,w,h)
        is_update_ok.append(False)  # 初始化
        tracker[n_obj].init(frame,bbox[n_obj])  # 使用视频的第一帧和边界框初始化跟踪器
        temp_p1 =(int(bbox[n_obj][0]), int(bbox[n_obj][1]))
        temp_p2 = (int(bbox[n_obj][0] + bbox[n_obj][2]), int(bbox[n_obj][1] + bbox[n_obj][3]))
        cv2.rectangle(frame_copy, temp_p1, temp_p2, (255, 0, 0), 2, 1)  # 在图上画框
        # cv2.imshow("window for select",frame_copy)
        n_obj += 1
        is_continue = input("Do you want to continue to select object?(y/n)\n")
        n = 0
        while True:
            flag = n % interval
            if flag == 0:
                cnt += 1
            # 读取新的视频流图像
            ok, frame = video.read()
            if not ok:
                print("Video comes to end or it has error to read video.")
                break
            image_data = frame.copy()  # 复制原始图像，用于画框显示跟踪结果
            w = frame.shape[1]
            h = frame.shape[0]

            # Start timer 记录开始时间
            timer = cv2.getTickCount()
            p1 = []  # 目标跟踪器返回图框的左上角点的信息列表
            p2 = []  # 目标跟踪器返回图框的右下角点的信息列表
            n_box = 0  # 图框数量

            for i in range(n_obj):
                is_update_ok[i], bbox[i] = tracker[i].update(frame)
                if is_update_ok[i]:
                    p1.append((int(bbox[i][0]), int(bbox[i][1])))
                    p2.append((int(bbox[i][0] + bbox[i][2]), int(bbox[i][1] + bbox[i][3])))
                    cv2.rectangle(image_data, p1[n_box], p2[n_box], (255, 0, 0), 2, 1)  # 在图上画框
                    n_box += 1

            # if (flag == 0):  # 如果图框个数为0，则跟踪失败
            #     # Tracking failure
            #     cv2.putText(image_data, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
            #                 (0, 0, 255), 2)
            cv2.putText(image_data, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50),
                        2)

            # Display FPS on frame 显示FPS
            # Calculate Frames per second (FPS) 计算FPS
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            cv2.putText(image_data, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50),
                        2)
            # Display result 显示跟踪结果
            cv2.imshow("Tracking", image_data)
            k = cv2.waitKey(1) & 0xff
            if k == 27: break
            # 或计数多少帧后退出，调试用
            n += 1
            if cnt > cnt_limit:
                print("exit")
                break