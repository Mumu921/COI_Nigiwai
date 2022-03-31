import random

from openvino.inference_engine import IECore
import numpy as np
import time
import cv2.cv2 as cv
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
        ih, iw, ic = frame.shape #获取视频中读出图片的尺寸:iw和ih,默认大小为iw=1920，ih=1088
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
          return res
          #return np.delete(res, 1) # 不明白为什么删除第一个数据


#-------------------------------------------#
# 行人追踪
# 使用行人再识别算法得到的[1, 256]结果，来判断前后两帧中的人是否属于同一人，然后进行追踪
#-------------------------------------------#
class Pedestrian_tracker:
    # 初始化database
    def __init__(self):
        self.identifysDS = None   # 检测到的人
        self.centerDS = []    # 检测到的人的bounding box的中心位置

    def getCenter(self, detection):
        x = detection[0] - detection[2]
        y = detection[1] - detection[3]
        return (x, y)

    # 计算检测出的bounding box的中心位置与centerDS中存在的bounding box的中心距离差
    def getDistance(self, detection, index):
        (x1, y1) = self.centerDS[index]
        x2 = detection[0] - detection[2]
        y2 = detection[1] - detection[3]
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
    def setIDs(self, identifys, detections):
        if (identifys.size == 0):
            return []
        # 如果identifyD
        if self.identifysDS is None:
            self.identifysDS = identifys
            for detection in detections:
                x = detection[0] - detection[2]
                y = detection[1] - detection[3]
                self.centerDS.append((x, y))

        print("input of bounding box: {} identifyDb:{}".format(len(identifys), len(self.identifysDS)))
        similaritys = self.cos_similarity(identifys, self.identifysDS)
        similaritys[np.isnan(similaritys)] = 0
        ids = np.nanargmax(similaritys, axis=1)

        # 更新id的策略:将新检测出的所有bounding box与id的数据库中的所有数据计算cos相似度，然后找到每个box对于数据库
        # 相似度最大值所在的位置,若该位置的相似度大于门限值，则证明新检测出的box在原来的id数据库中，并更新数据库中该id的信息([1，256])；
        # 若相似度小于门限值，证明新检测出的box不在原来的id数据库中，则将该新检测出的box的信息添加到id数据库中
        for i, similarity in enumerate(similaritys):
            detectionId = ids[i]
            d = self.getDistance(detections[i], detectionId)
            print("detectionId:{} cos Similarity:{} distance:{}".format(detectionId, similarity[detectionId], d))
            # 相似度大于0.95，并且没有重合时，更新判别条件,如果新识别到的bounding box框在原来的数据库中，则更新该box在数据库中相对应位置的信息
            if (similarity[detectionId] > 0.95):
                if (self.isOverlap(detections, i) == False):
                    self.identifysDS[detectionId] = identifys[i]
                    print("similarity is over than 0.95")
             # 相似度低于0.5时，更新到identifyDS中，即添加一个新成员
            elif (similarity[detectionId] < 0.5):
                if (d > 100):
                    print("distance:{} similarity:{}".format(d, similarity[detectionId]))
                    self.identifysDS = np.vstack((self.identifysDS, identifys[i])) # 新id添加到数据库中
                    self.centerDS.append(self.getCenter(detections[i]))
                    ids[i] = len(self.identifysDS) - 1
                    print("> append DB size:{}".format(len(self.identifysDS)))

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
        print(ids)
        return ids




if __name__ == "__main__":
    pedestrian_detection = Pedestrian_Detection() #该语句之前必须要有缩进，原因不明
    pedestrianreid = pedestrian_Reid()
    pedestriantacker = Pedestrian_tracker()

SCALE = 0.3
track_number = 60
box_colors = []   # 生成bounding box的颜色库
for i in range(track_number):
    b = random.randint(0, 255)  # 随机生成bgr值
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    box_colors.append((b, g, r))

# cap = cv.VideoCapture("C:/image/video/COI-video/20210928/mov_area1_2021-09-10_14-00-01_600.mp4")  # 视频文件读入
cap = cv.VideoCapture("C:/image/video/2.mp4")  # 视频文件读入
# cap = cv.VideoCapture("C:/image/video/vibration.mp4")  # 视频文件读入
while True:

    ret, frame = cap.read()
    if ret is not True:
        break
    detections, infer_time = pedestrian_detection.infer(frame)
    print(detections)
    if (len(detections) > 0):
        print("detection result")
        for detection in detections:
            xmin = detection[0]
            ymin = detection[1]
            xmax = detection[2]
            ymax = detection[3]
            conf = detection[4]
           # cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2, 8)  # 人物检测框
            #cv.putText(frame, str(conf), (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 225), 1)

    #cv.putText(frame, "infer time(ms): %.3f, FPS:%.2f" % (infer_time, 1 / (infer_time + 0.000000001)), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2, 8)
    identifys = np.zeros((len(detections), 256))
    for i, detection in enumerate(detections):
        img = frame[detection[1]:detection[3], detection[0]:detection[2]]
        h, w = img.shape[:2]
        if (h == 0 or w == 0):  # 若为检测到图像，则进行下一次循环
            continue
        identifys[i] = pedestrianreid.reid_infer(img)
        print("shape of every result of reid:{}".format(identifys[i].shape))
        #print("result of reid:{}".format(identifys[i]))
    print("shape of result of reid:{}".format(identifys.shape))

    # 获得行人的id
    ids = pedestriantacker.setIDs(identifys, detections)

    # 绘制行人检测框和id
    for i, detection in enumerate(detections):
        if (ids[i] != -1):
            color = box_colors[int(ids[i])]
            frame = cv.rectangle(frame, (detection[0], detection[1]), (detection[2], detection[3]), color, int(50 * 0.3))
            frame = cv.putText(frame, str(ids[i]), (detection[0], detection[1]), cv.FONT_HERSHEY_PLAIN, int(50 * 0.3),
                                color, int(30 * 0.3), cv.LINE_AA)

    # 缩小画面
    h, w = frame.shape[:2]
    frame = cv.resize(frame, ((int(w * SCALE), int(h * SCALE))))
    cv.namedWindow('pedestrian-detection', cv.WINDOW_NORMAL)
    cv.imshow("pedestrian-detection", frame)
    c = cv.waitKey(50)
    if c == 27:
        break
cap.release()
cv.destroyAllWindows()
