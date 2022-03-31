## 该程序用于旋转图片的角度
import cv2.cv2 as cv
import numpy as np



# 读取视频中的图片并保存
def Picture_in_Video():
    cap1 = cv.VideoCapture("C:/image/video/COI-video/20210928/mov_area1_2021-09-10_14-00-01_600_2.mp4")  # 视频文件读入
    cap2 = cv.VideoCapture("C:/image/video/COI-video/20210928/mov_area1_2021-09-10_14-00-01_600.mp4")  # 视频文件读入
    save_path = r"C:/image/video/COI-video/20210928/picture/"
    c = 0
    while(1):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        cv.imshow("Picture1",frame1)
        cv.imshow("Picture2", frame2)
        cv.imwrite(save_path + 'left-image'+ str(c) + '.jpg', frame1)
        cv.imwrite(save_path + 'right-image' + str(c) + '.jpg', frame2)
        c = c+1
        c1 = cv.waitKey(100)
        if c1 == 27:
            break
    cap1.release()
    cap2.release()
    cv.destroyAllWindows()
#

# 该函数用于找到图片中的参考点
def point_in_picture(points_list, img):
    # points_list = [(960, 544), (1550, 460), (900, 460), (1730, 470)]
    points_list = points_list
    for point in points_list:
        cv.circle(img, point, 3, (0, 0, 255), 5)

# 通过鼠标点击确定图片上的参考点
def position_found_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        xy = str(x) + "," + str(y)
        print("coordinates at click:x,y:{},{}".format(x,y))
        cv.circle(img, (x,y), 1 , (255,0,0), thickness=-1)
        cv.putText(img, xy, (x,y),cv.FONT_HERSHEY_PLAIN, 0.8, (0,0,255), thickness=1)
        cv.imshow("original",img)

# 在图片上绘制线段
def line_in_picture(img, points, color):
    plot_point = []
    for i, point in enumerate(points):
        x = int(point[0])
        y = int(point[1])
        pointinlist = (x,y)
        plot_point.append(pointinlist)
    (tl, tr, br, bl) = plot_point
    print("point to be plotted {}".format(plot_point))
    # # 上左点
    # point_start = (1500,140)
    # point_end = (1550,460)
    # color = (0,0,255)
    # # 上右点
    # point_start2 = (1700, 140)
    # point_end2 = (1730, 470)
    # color2 = (0, 255, 0)
    # # 下右点
    # point_start3 = (1550, 460)
    # point_end3 = (1730, 470)
    # color3 = (255, 0, 0)
    # # 下左点
    # point_start3 = (1550, 460)
    # point_end3 = (1730, 470)
    # color3 = (255, 0, 0)
    thickness = 2
    # 绘制四条边
    img = cv.line(img, tl, tr, color, thickness)
    img = cv.line(img, tl, bl, color, thickness)
    img = cv.line(img, tr, br, color, thickness)
    img = cv.line(img, bl, br, color, thickness)
    print("line in picture")

# 计算输入点四边的长度，输入点的顺序为:上左(tl)，上右(tr)，下右(br)，下左(bl)
def Four_points_transform(img, points):
    #计算上下边的边长
    (tl,tr,br,bl) = points
    width_top = np.sqrt(((tl[0]-tr[0]) ** 2) + ((tl[1]-tr[1]) ** 2))
    width_bottom = np.sqrt(((bl[0] - br[0]) ** 2) + ((bl[1] - br[1]) ** 2))
    maxWidth = max(int(width_top), int(width_bottom))
    # 计算上左右的边长
    height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    maxHeight = max(int(height_left), int(height_right))
    print("maxWidth: {}, maxHeight: {}".format(maxWidth, maxHeight))
    # 输入点经过变换之后的四点位置
    maxHeight, maxWidth= img.shape[:2]
    #points_transform = np.array([(0, 0), (maxWidth-1, 0), (maxWidth-1, maxHeight-1), (0, maxHeight-1)], dtype="float32")
    points_transform = np.array([(1550, 140), (1700, 140), (1730, 470), (1550, 460)], dtype="float32") # 门的位置坐标
    # 计算转换矩阵
    Matrix = cv.getPerspectiveTransform(points, points_transform)
    # print("matrix of transform {}".format(Matrix))
    # 实现透视转换变换
    img_process = cv.warpPerspective(img, Matrix, (maxWidth, maxHeight))
    # img_process = cv.warpPerspective(img, Matrix, (1920, 1088))
    return img_process

if __name__ == "__main__":
    # 读取视频中的图片并保存
    # Picture_in_Video()
    # 读取需要透视变换的照片
    img = cv.imread("C:/image/video/COI-video/20210928/picture/left-image50.jpg")
    # 打印读入图片的长宽
    img_height, img_width = img.shape[:2]
    print("width:{}, height:{}".format(img_width, img_height))
    # 显示原图
    cv.namedWindow('original', cv.WINDOW_NORMAL)
    # point_in_picture(img)
    # 在原图中找出门的位置
    # door_points = np.array([(1500, 140), (1700, 140), (1730, 470), (1550, 460)], dtype="float32")
    # line_in_picture(img, door_points, color=(0,0,255))
    #cv.imshow("original", img)
    # 在原图中找出地板的位置
    # tile_points = np.array([(1500, 140), (1700, 140), (1730, 470), (1550, 460)],dtype="float32")
    # tile_points2 = np.array([(1500, 140), (1730, 140), (1730, 470), (1500, 470)],dtype="float32")
    # line_in_picture(img, tile_points, color=(0, 0, 255))
    # 在原图中找出需要变化图片的位置
    #picture_points = np.array([(50, 50), (1890, 30), (1860, 1030), (20, 1070)], dtype="float32")
    #line_in_picture(img, picture_points, color=(0, 0, 255))
    # 在原图中找出需要透视变换区间范围
    left_area_points = np.array([(865, 806), (953, 864), (815, 984), (732, 907)], dtype="float32")
    # right_area_points = np.array([(40, 230), (1900, 190), (1870, 1080), (20, 1080)], dtype="float32")
    #line_in_picture(img, right_area_points, color=(0,0,255))
    # 输入原图中需要透视变换的四点
    points_list = left_area_points
    # 在图中显示需要变换的四点
    transform_points_list = [(865, 806), (953, 864), (815, 984), (732, 907)]
    point_in_picture(transform_points_list, img)
    img_process = Four_points_transform(img, points_list)
    # 绘制原图
    cv.namedWindow('original', cv.WINDOW_NORMAL)

    # 使用鼠标找目标点
    cv.setMouseCallback("original", position_found_LBUTTONDOWN)
    cv.imshow("original", img)
    # # 显示透视变换之后的图
    cv.namedWindow('process', cv.WINDOW_NORMAL)
    cv.imshow("process", img_process)
    #cv.namedWindow('process2', cv.WINDOW_AUTOSIZE)
    #cv.imshow("process2", img_process2)

    while(True):
        try:
            cv.waitKey(100)
        except Exception:
            cv.destroyAllWindows("original")
    cv.waitKey(0)
    cv.destroyAllWindows()
