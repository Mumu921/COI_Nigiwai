import numpy as np
import pandas as pd

## Function for model evaluation


# 计算点到线段的距离
def point_distance_line(point,line_point1,line_point2):
    # 计算向量
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1,vec2)) / np.linalg.norm(line_point1-line_point2)
    return distance

# 在相机画面中找到规定的测试范围并计算两条线段之间的距离
def start_end_area(camera1_line1,camera1_line2,camera2_line1,camera2_line2):
    # 从line1中选择一个点来计算到line2的距离，从而得到两条线段之间的距离
    camera1_area_width = point_distance_line(camera1_line1[0],camera1_line2[0],camera1_line2[1])
    camera2_area_width = point_distance_line(camera2_line1[0], camera2_line2[0], camera2_line2[1])
    return camera1_area_width,camera2_area_width


# 判断一个点是否通过规定的测试范围
def point_in_area(point, line1, line2):
    area_width = point_distance_line(line1[0], line2[0], line2[1]) # 规定范围的两条线段之间的距离
    distance1 = point_distance_line(point, line1[0], line1[1])  # 任意一点到线段1的距离
    distance2 = point_distance_line(point, line2[0], line2[1])  # 任意一点到线段2的距离
    if distance1 <= area_width and distance2 <= area_width:
        return True
    else:
        return False


# 查找在两个摄像机中都出现的行人
def People_in_camera(filepath, camera1_line1, camera1_line2, camera2_line1, camera2_line2, ID_in_camera1, ID_in_camera2, ID_in_camera1_camera2):
    track_file = pd.read_csv(filepath, header = 0 )
    # print(track_file)
    ID_summary = np.unique(track_file["ID"], return_counts = True)
    ID_type = ID_summary[0]   # 读取的csv文件中有多少ID编号，结果是列表
    ID_num = ID_summary[1]   # 每种编号有多少个，结果是列表
    # print("ID:{}".format(ID_type))
    # print("ID_num:{}".format(ID_num))
    # ID_index_filter = np.where(ID_num > 20)
    # print("ID_index_filter:{}".format(ID_index_filter))
    # for i, index in enumerate(ID_index_filter):
    #     print("index: {}".format(index))
    #     track_select = track_file["ID" == index]
    for i, index in enumerate(ID_type):
        ID_selected = track_file.loc[track_file['ID'] == index, :]
        # print("ID_selected {}".format(ID_selected))
        ID_in_camera = np.unique(ID_selected['Camera_index'], return_counts=True)   # ID编号在camera1，2中出现的次数
        # print("ID_selected {}".format(ID_in_camera))
        ID_in_camera_type = ID_in_camera[0]
        ID_in_camera_num = ID_in_camera[1]
        # print("len of ID_in_camera_type:{}".format(len(ID_in_camera_type)))
        # 判断该ID号在camera1和camera2中是否为有效番号
        if len(ID_in_camera_type) == 2:     # 找到同时存在camera1，2中的ID，并保存
            if ID_in_camera_num[0] > 20 and ID_in_camera_num[1] > 20:
                camera1_inf = ID_selected.loc[ID_selected["Camera_index"] == ID_in_camera_type[0], :]
                camera2_inf = ID_selected.loc[ID_selected["Camera_index"] == ID_in_camera_type[1], :]
                point_camera1_count = 0
                point_camera2_count = 0
                for i in range(len(camera1_inf)):
                    inf = camera1_inf.iloc[i]
                    x = inf[1]
                    y = inf[2]
                    point = np.array([x, y])
                    if point_in_area(point, camera1_line1, camera1_line2):
                        point_camera1_count += 1
                for i in range(len(camera2_inf)):
                    inf = camera2_inf.iloc[i]
                    x = inf[1]
                    y = inf[2]
                    point = np.array([x, y])
                    if point_in_area(point, camera2_line1, camera2_line2):
                        point_camera2_count += 1
                if point_camera1_count >= 5 and point_camera2_count >= 5:
                    ID_in_camera1_camera2.append(index)

        if len(ID_in_camera_type) == 1:
            # 判断该ID号在camera1中是否为有效番号
            if ID_in_camera_type[0] == 1 and ID_in_camera_num[0] > 20:
                point_count = 0
                for i in range(len(ID_selected)):
                    inf = ID_selected.iloc[i]
                    x = inf[1]
                    y = inf[2]
                    point = np.array([x, y])
                    if point_in_area(point, camera1_line1, camera1_line2):
                        point_count += 1
                if point_count >= 5:
                    ID_in_camera1.append(index)

            # 判断该ID号在camera2中是否为有效番号
            if ID_in_camera_type[0] == 2 and ID_in_camera_num[0] > 20:
                point_count = 0
                for i in range(len(ID_selected)):
                    # print("i is {}".format(i))
                    # print("len is {}".format(len(ID_selected)))
                    inf = ID_selected.iloc[i]
                    x = inf[1]
                    y = inf[2]
                    point = np.array([x,y])
                    if point_in_area(point, camera2_line1, camera2_line2):
                        point_count += 1
                if point_count >= 5:
                    ID_in_camera2.append(index)

    return ID_in_camera1, ID_in_camera2, ID_in_camera1_camera2


filepath = r"C:\Users\lin18\PycharmProjects\COI_Nigiwai\Track of Pedestrian.csv"

# 规定检测范围区域，该区域为line1和line2之间的距离
camera1_line1 = np.array(([300, 1088], [1350, 450]))
camera1_line2 = np.array(([500, 1088], [1550, 450]))
camera2_line1 = np.array(([550, 650], [1100, 1088]))
camera2_line2 = np.array(([750, 650], [1350, 1088]))
camera1_area_width,camera2_area_width = start_end_area(camera1_line1, camera1_line2, camera2_line1, camera2_line2)
print('area1:{},area2:{}'.format(camera1_area_width, camera2_area_width))

ID_in_camera1 = []
ID_in_camera2 = []
ID_in_camera1_camera2 = []

ID_in_camera1, ID_in_camera2, ID_in_camera1_camera2 = People_in_camera(filepath, camera1_line1, camera1_line2, camera2_line1, camera2_line2, ID_in_camera1, ID_in_camera2, ID_in_camera1_camera2)
print(ID_in_camera1, ID_in_camera2, ID_in_camera1_camera2 )
# People_in_two_camera(filepath)