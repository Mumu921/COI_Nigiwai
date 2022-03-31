## 实现了两个摄像机之间的行人再识别，但是没有记录行人来自与哪个摄像机。

import random
from openvino.inference_engine import IECore
import numpy as np
import time
import cv2.cv2 as cv
import pandas as pd
##---------------------------------------------------------------------##
# 将图片输入该函数，可以得到图片中检测出的人的bounding box的位置以及置信度
##---------------------------------------------------------------------##
# 行人检测
class Pedestrian_Detection():

    def __init__(self):
        ie = IECore()
         # 行人检测模型的xml和bin文件载入
        model_xml = "C:/openvino_download_model/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml"
        model_bin = "C:/openvino_download_model/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.bin"
        # 载入行人检测模型
        self.net = ie.read_network(model=model_xml, weights=model_bin)
        self.input_blob = next(iter(self.net.input_info))  # 读取模型结构的接口
        self.out_blob = next(iter(self.net.outputs))
        n, c, h, w = self.net.input_info[self.input_blob].input_data.shape  # 输入模型中的图片的尺寸：NCHW
        self.input_h = h
        self.input_w = w
        self.exec_net = ie.load_network(network=self.net, device_name="CPU")  # 载入模型的网络

    def infer(self, frame):
        image = cv.resize(frame, (self.input_w, self.input_h))  # 将输入的图片的宽和高转换为推理模型要求的尺寸，w：宽；h：高
        image = image.transpose(2, 0, 1) # VidepCapyture读出的图片为HWC格式，需要转换为模型要求的CHW格式
        inf_start = time.time()
        res = self.exec_net.infer(inputs={self.input_blob: [image]})  # image的格式为三维的CHW，该处理之后变成四维就可以被模型接口读取
        inf_end = time.time() - inf_start
        infer_time = inf_end * 1000
        print("infer time(ms): %.3f" % (inf_end * 1000))
        person_boundingboxs = []  # 存储图片中推断出来的bounding box的信息，包括xmin,ymin,xmax,ymax和conf
        ih, iw, ic = frame.shape #获取视频中读出图片的尺寸:iw和ih, 默认大小为iw=1920，ih=1088
        res = res[self.out_blob]  # 获得输出结果，包含很多的结果，按要求找到想要的结果
        for obj in res[0][0]:  # res[0]表示图片中存在的检测对象，res[0][0]代表每个对象中的7个参数
            if obj[2] > 0.25: # 设置置信度score
                ## index = int(obj[1]) - 1  # 给labels的index
                xmin = int(obj[3] * iw)
                ymin = int(obj[4] * ih)
                xmax = int(obj[5] * iw)
                ymax = int(obj[6] * ih)
                conf = obj[2]
                person_boundingboxs.append([xmin,ymin,xmax,ymax,conf])
        return person_boundingboxs, infer_time

#-------------------------------------------#
#行人再识别模型
# 模型参考URL:file:///C:/Program%20Files%20(x86)/Intel/openvino_2020.4.287/deployment_tools/open_model_zoo/models/intel/person-reidentification-retail-0265/description/person-reidentification-retail-0265.html
# 模型的input: 检测出来的bounding box
# 模型的output：bounding box的一维向量，[1,256]
#-------------------------------------------#
class pedestrian_Reid():

      def __init__(self):
          reid_ie = IECore()
          # 行人再识别模型
          reid_xml = "C:/openvino_download_model/intel/person-reidentification-retail-0265/FP32/person-reidentification-retail-0265.xml"
          reid_bin = "C:/openvino_download_model/intel/person-reidentification-retail-0265/FP32/person-reidentification-retail-0265.bin"

          # 载入行人检测模型
          self.reid_net = reid_ie.read_network(model=reid_xml, weights=reid_bin)
          self.reid_input_blob = next(iter(self.reid_net.input_info))  # 读取模型结构的接口
          self.reid_out_blob = next(iter(self.reid_net.outputs))

          reid_n, reid_c, reid_h, reid_w = self.reid_net.input_info[self.reid_input_blob].input_data.shape  # NCHW
          print(reid_n, reid_c, reid_h, reid_w)
          self.reid_input_h = reid_h
          self.reid_input_w = reid_w
          self.reid_exec_net = reid_ie.load_network(network=self.reid_net, device_name="CPU")  # 载入模型的网络

      def reid_infer(self, boundingbox):
          image = cv.resize(boundingbox, (self.reid_input_w, self.reid_input_h))  # 将输入的图片的宽和高转换为推理模型要求的尺寸，w：宽；h：高
          image = image.transpose(2, 0, 1)  # VidepCapyture读出的图片为HWC格式，需要转换为模型要求的CHW格式
          res = self.reid_exec_net.infer(inputs={self.reid_input_blob: [image]})  # image的格式为三维的CHW，该处理之后变成四维就可以被模型接口读取
          res = res[self.reid_out_blob]  # [1，256]结果
          # print("res is:{}".format(res))
          return res

          #return np.delete(res, 1) # 不明白为什么删除第一个数据


#-------------------------------------------#
# 行人追踪
# 使用行人再识别算法得到的[1, 256]结果，来判断前后两帧中的人是否属于同一人，然后进行追踪
#-------------------------------------------#
class Pedestrian_tracker:
    # 初始化database
    def __init__(self):
        self.identifysDS_centerDS_library = [[] for _ in range(4)] # 记录多个视频文件的identifysDS和centerDS信息
        self.identifysDS_centerDS_library[0] = None # self.identifysDS_cam1 = self.identifysDS_centerDS_library[0], camera1中检测到的人的[1，256]数据
        # self.centerDS_cam1 = identifysDS_centerDS_library[1]    # camera1中检测到的人的bounding box的中心位置
        self.identifysDS_centerDS_library[2] = None  # self.identifysDS_cam2 = self.identifysDS_centerDS_library[2], camera1中检测到的人的[1，256]数据
        # self.centerDS_cam2 = identifysDS_centerDS_library[3]  # camera2中检测到的人的bounding box的中心位置
        self.identifysDS = None  # 多个视频文件中的identifys信息汇总
        self.centerDS = []  # 带标签boundingbox的中心位置

    def getCenter(self, detection): # detection参数为从每张图片中检测出的一个bounding box
        x = (detection[2] - detection[0])/2
        y = (detection[3] - detection[1])/2
        return (x, y)

    # 计算检测出的bounding box的中心位置与centerDS中存在的bounding box的中心距离差
    def getDistance(self, detection, index, camera_index):
        library_index = 2 * camera_index - 1   # 根据视频的编号查找到对应的library中用于记录视频位置的列表
        print("library is ok")
        print("camera_index is:{} centerDS is: {}".format(camera_index, self.identifysDS_centerDS_library[library_index]))
        (x1, y1) = self.identifysDS_centerDS_library[library_index][index]
        print("x1 y1 read is ok")
        x2 = (detection[2] - detection[0])/2
        y2 = (detection[3] - detection[1])/2
        centerDS_center = np.array([x1, y1])
        detection_center = np.array([x2, y2])
        distance = centerDS_center - detection_center
        return np.linalg.norm(distance)   # 返回两个向量之间的距离


    # 判断两个bounding box是否重合
    def isOverlap(self, detections, index):
        [xmin, ymin, xmax, ymax,confidence] = detections[index]
        for i, detection in enumerate(detections):
            if (index == i):
                continue
            if (max(detection[0], xmin) <= min(detection[2], xmax) and max(detection[1], ymin) <= min(detection[3], ymax)):
                return True
        return False


    # 判断两个向量的cos相似度
    def cos_similarity(self, X, Y):
        m = X.shape[0]
        Y = Y.T
        return np.dot(X, Y)/(np.linalg.norm(X.T, axis= 0).reshape(m, 1) * np.linalg.norm(Y, axis=0))

    # 赋予不重复的bounding box编号
    # 输入:identifys: 视频中读取的每张图片中每个boundingbox的[1，256]数据;  detections:每张图片中的所有boundingbox
    def setIDs(self, identifys, detections, camera_index):
        if (identifys.size == 0):
            return []
        # 如果identifyD
        identifys_lib_index = 2 * (camera_index - 1)
        # identifys_lib_index = 0
        center_lib_index = 2 * camera_index - 1
        if self.identifysDS_centerDS_library[identifys_lib_index] is None:
            self.identifysDS_centerDS_library[identifys_lib_index] = identifys
            for detection in detections:
                x = (detection[2] - detection[0])/2
                y = (detection[3] - detection[1])/2
                print("first centerDS x:{}, y:{} in camera{}".format(x,y,camera_index))
                self.identifysDS_centerDS_library[center_lib_index].append((x, y))

        print("input of bounding box: {} data from camera:{} length of identifyDb:{}".format(len(identifys), camera_index, len(self.identifysDS_centerDS_library[identifys_lib_index])))

        similaritys = self.cos_similarity(identifys, self.identifysDS_centerDS_library[identifys_lib_index])
        similaritys[np.isnan(similaritys)] = 0
        print("similaritys is {}".format(similaritys))
        ids = np.nanargmax(similaritys, axis=1)
        print("the ids is {}".format(ids))

        # 更新id的策略:将新检测出的所有bounding box与id的数据库中的所有数据计算cos相似度，然后找到每个box对于数据库
        # 相似度最大值所在的位置,若该位置的相似度大于门限值，则证明新检测出的box在原来的id数据库中，并更新数据库中该id的信息([1，256])；
        # 若相似度小于门限值，证明新检测出的box不在原来的id数据库中，则将该新检测出的box的信息添加到id数据库中
        for i, similarity in enumerate(similaritys):
            print("the index is: {}, the similarity is: {}".format(i, similarity))
            detectionId = ids[i]
            d = self.getDistance(detections[i], detectionId, camera_index)
            print("detectionId:{} cos Similarity:{} distance:{}".format(detectionId, similarity[detectionId], d))
            # 相似度大于0.95，并且没有重合时，更新判别条件[1,256],如果新识别到的bounding box框在原来的数据库中，则更新该box在数据库中相对应位置的信息
            if (similarity[detectionId] > 0.95):
                if (self.isOverlap(detections, i) == False):
                    self.identifysDS_centerDS_library[identifys_lib_index][detectionId] = identifys[i]  # 更新识别信息[1，256]
                    self.identifysDS_centerDS_library[center_lib_index][detectionId] = self.getCenter(detections[i])  # 更新centerDS中的位置信息
             # 相似度低于0.5时，更新到identifyDS中，即添加一个新成员
            elif (similarity[detectionId] < 0.5):
                if (d > 30):
                    print("distance:{} similarity:{}".format(d, similarity[detectionId]))
                    self.identifysDS_centerDS_library[identifys_lib_index] = np.vstack((self.identifysDS_centerDS_library[identifys_lib_index], identifys[i])) # 新id添加到数据库中
                    self.identifysDS_centerDS_library[center_lib_index].append(self.getCenter(detections[i]))
                    ids[i] = len(self.identifysDS_centerDS_library[identifys_lib_index]) - 1
                    # print("append DB size:{}".format(len(self.identifysDS)))
        print(ids)
        # 有重复编号时，去掉信赖度低的编号
        for i, a in enumerate(ids):
            for e, b in enumerate(ids):
                if (e == i):
                    continue
                if (a == b):
                    if (similarity[a] > similarity[b]):
                        ids[i] = -1
                    else:
                        ids[e] = -1
        print("updated ids:{}".format(ids))
        return ids




if __name__ == "__main__":
    pedestrian_detection = Pedestrian_Detection() #该语句之前必须要有缩进，原因不明
    pedestrianreid = pedestrian_Reid()
    pedestriantacker = Pedestrian_tracker()
    pedestriantacker_cap2 = Pedestrian_tracker()

camera1_finish_count = 0
camera2_finish_count = 0
frame_number = 0
SCALE = 0.3
track_number = 200
box_colors = []   # 生成bounding box的颜色库
Pedestrian_location = [[] for _ in range(track_number)]  # 保存检测到的行人的中心的坐标
Pedestrian_location_cap2 = [[] for _ in range(track_number)]  # 保存camera2中检测到的行人的中心的坐标
for i in range(track_number):
    b = random.randint(0, 255)  # 随机生成bgr值
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    box_colors.append((b, g, r))
#
# cap1 = cv.VideoCapture("C:/image/video/COI-video/20210928/mov_area1_2021-09-10_14-00-01_600_2.mp4")  # 视频文件读入
# cap2 = cv.VideoCapture("C:/image/video/COI-video/20210928/mov_area1_2021-09-10_14-00-01_600.mp4")  # 视频文件读入

while True:

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if (ret1 is not True) and (ret2 is not True):
        break
    detections_cap1, infer_time_1 = pedestrian_detection.infer(frame1)
    detections_cap2, infer_time_2 = pedestrian_detection.infer(frame2)

    print('bounding box detected from model in camera 1:{}'.format(detections_cap1))
    print('bounding box detected from model in camera 2:{}'.format(detections_cap2))
    # if (len(detections) > 0):
    #     print("detection result")
    #     for detection in detections:

    #         xmin = detection[0]
    #         ymin = detection[1]
    #         xmax = detection[2]
    #         ymax = detection[3]
    #         conf = detection[4]
           # cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2, 8)  # 人物检测框
            #cv.putText(frame, str(conf), (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 225), 1)

    # camera1中的行人附上id号并绘制移动线路
    #cv.putText(frame, "infer time(ms): %.3f, FPS:%.2f" % (infer_time, 1 / (infer_time + 0.000000001)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2, 8)
    identifys_cap1 = np.zeros((len(detections_cap1), 256))
    for i, detection in enumerate(detections_cap1):
        img_cap1 = frame1[detection[1]:detection[3], detection[0]:detection[2]]
        h, w = img_cap1.shape[:2]
        if (h == 0 or w == 0):  # 若为检测到图像，则进行下一次循环
            continue
        identifys_cap1[i] = pedestrianreid.reid_infer(img_cap1) # 推断得到每个boundingbox的[1，256]数据，用于每个boundingbox的相似度计算
        print("shape of every result of reid:{}".format(identifys_cap1[i].shape))
        #print("result of reid:{}".format(identifys[i]))
    print("shape of result of reid:{}".format(identifys_cap1.shape))

    # 获得camera1中行人的id
    ids_cap1 = pedestriantacker.setIDs(identifys_cap1, detections_cap1, 1) # 处理第一个视频文件
    # 绘制行人检测框和id
    for i, detection in enumerate(detections_cap1):
        if (ids_cap1[i] != -1):
            color = box_colors[int(ids_cap1[i])]
            frame1 = cv.rectangle(frame1, (detection[0], detection[1]), (detection[2], detection[3]), color, int(50 * 0.1))
            frame1 = cv.putText(frame1, str(ids_cap1[i]), (detection[0], detection[1]), cv.FONT_HERSHEY_PLAIN, int(50 * 0.1),
                                color, int(30 * 0.1), cv.LINE_AA)
            # 记录标有id的行人的中心位置
            PedestrianID = ids_cap1[i]
            center_x_cap1 = int((detection[2] - detection[0]) / 2) + detection[0]
            center_y_cap1 = int((detection[3] - detection[1]) / 2) + detection[1]
            Pedestrian_location[PedestrianID].append((center_x_cap1, center_y_cap1))
            # 绘制出带有ID的行人的中心位置
            # frame = cv.circle(frame, (center_x, center_y), 2, color, 4)
    #print("the location of pedestrian is :{}".format(Pedestrian_location))
    # 绘制运动轨迹
    thickness = 6
    for i, location_list in enumerate(Pedestrian_location):
        for m in range(1, len(location_list)):
            if location_list[m-1] is None or location_list[m] is None:
                continue
            color = box_colors[i]
            # print("color is:".format(color))
            distance = np.array(location_list[m]) - np.array(location_list[m-1]) # 如果出现行人位置判别偏差较大，则将异常点人为修正到正确范围
            point_distance = np.linalg.norm(distance)
            if point_distance > 10:
                tem_point = np.array(location_list[m-1]) + np.array((5,5))
                tem_point = tuple(tem_point)
                frame1 = cv.line(frame1, location_list[m-1], tem_point, color, thickness)
            else:
                frame1 = cv.line(frame1, location_list[m - 1], location_list[m], color, thickness)
    camera1_finish_count += 1
    print("camera 1 is finished {} times".format(camera1_finish_count))

    #---------------------------------------------------------#
# 视频2的行人检测
# ---------------------------------------------------------#
    # camera2中的行人附上id号并绘制移动线路
    identifys_cap2 = np.zeros((len(detections_cap2), 256))
    for i, detection in enumerate(detections_cap2):
        img_cap2 = frame2[detection[1]:detection[3], detection[0]:detection[2]]
        h, w = img_cap2.shape[:2]
        if (h == 0 or w == 0):  # 若为检测到图像，则进行下一次循环
            continue
        identifys_cap2[i] = pedestrianreid.reid_infer(img_cap2) # 推断得到每个boundingbox的[1，256]数据，用于每个boundingbox的相似度计算
        print("shape of every result of reid:{}".format(identifys_cap2[i].shape))
        #print("result of reid:{}".format(identifys[i]))
    print("shape of result of reid:{}".format(identifys_cap2.shape))

    # 获得camera2中行人的id
    ids_cap2 = pedestriantacker.setIDs(identifys_cap2, detections_cap2, 2)  # 处理第一个视频文件
    # 绘制行人检测框和id
    for i, detection in enumerate(detections_cap2):
        if (ids_cap2[i] != -1):
            color = box_colors[int(ids_cap2[i])]
            frame2 = cv.rectangle(frame2, (detection[0], detection[1]), (detection[2], detection[3]), color,
                                  int(50 * 0.1))
            frame2 = cv.putText(frame2, str(ids_cap2[i]), (detection[0], detection[1]), cv.FONT_HERSHEY_PLAIN,
                                int(50 * 0.1),
                                color, int(30 * 0.1), cv.LINE_AA)
            # 记录标有id的行人的中心位置
            PedestrianID = ids_cap2[i]
            center_x_cap2 = int((detection[2] - detection[0]) / 2) + detection[0]
            center_y_cap2 = int((detection[3] - detection[1]) / 2) + detection[1]
            Pedestrian_location_cap2[PedestrianID].append((center_x_cap2, center_y_cap2))
            # 绘制出带有ID的行人的中心位置
           # frame2 = cv.circle(frame2, (center_x_cap2, center_y_cap2), 2, color, 4)
    # print("the location of pedestrian is :{}".format(Pedestrian_location))
    # 绘制运动轨迹
    thickness = 6
    for i, location_list in enumerate(Pedestrian_location_cap2):
        for m in range(1, len(location_list)):
            if location_list[m - 1] is None or location_list[m] is None:
                continue
            color = box_colors[i]
            # print("color is:".format(color))
            distance = np.array(location_list[m]) - np.array(location_list[m - 1])  # 如果出现行人位置判别偏差较大，则将异常点人为修正到正确范围
            point_distance = np.linalg.norm(distance)
            if point_distance > 10:
                tem_point = np.array(location_list[m - 1]) + np.array((5, 5))
                tem_point = tuple(tem_point)
                frame2 = cv.line(frame2, location_list[m - 1], tem_point, color, thickness)
            else:
                frame2 = cv.line(frame2, location_list[m - 1], location_list[m], color, thickness)
    camera2_finish_count += 1
    print("camera 2 is finished {} times".format(camera2_finish_count))

    # 缩小画面
    h, w = frame1.shape[:2]
    frame1 = cv.resize(frame1, ((int(w * SCALE), int(h * SCALE))))
    frame2 = cv.resize(frame2, ((int(w * SCALE), int(h * SCALE))))
    cv.namedWindow('pedestrian-detection', cv.WINDOW_NORMAL)
    cv.imshow("pedestrian-detection", frame1)
    cv.namedWindow('pedestrian-detection-camera2', cv.WINDOW_NORMAL)
    cv.imshow("pedestrian-detection-camera2", frame2)
    c = cv.waitKey(50)
    if c == 27:
        break
    # frame_number += 1  # 记录检测的视频中的帧数
cap1.release()
cap2.release()
cv.destroyAllWindows()
print("dsf")
#print("the frame in video is:{}".format(frame_number))
# 追踪到的行人轨迹的位置坐标保存
track_location = pd.DataFrame(Pedestrian_location)
track_location.to_csv("Location of Pedestrian track.csv")