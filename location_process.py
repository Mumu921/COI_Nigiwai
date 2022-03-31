##该程序将得到的位置信息转换为理想的格式

import pandas as pd
# location_info = pd.read_csv("./Location of Pedestrian track.csv")
# print(location_info)
# # 读取每一行数据
# a = location_info.iloc[1,:][1]
# a_1 = a[1:(len(a)-1)]
# print("delete brackets:{}".format(a_1))
# print("location is {}".format(a))
# print("length :{}".format(len(a)))
# a_list = a_1.split(", ")
# print("length2 :{}".format(len(a_list)))
# for i in range(len(a_list)):
#     num = i
#     print("contents in a_list:{}".format(a_list[num]))
# print("processed a:{}".format(a_list))

location_log = pd.DataFrame(columns=["ID", "X", "Y", "Camera_index", "Confidence"])
print(location_log)
b = []
P = [[] for _ in range(5)]
p1 = [(12,154,1,0.365),(484,13,1,0.57)]
p2 = [(47,545,2,0.674)]
p3 = [(123,21,3,0.578),(124,12,3,0.987),(33,65,3,0.645)]
P[0] = p1
P[1] = p2
P[2] = p3
print("list p is:{}".format(P))
for i, pedestrian_location in enumerate(P):
    location = pedestrian_location
    for j, coordinate in enumerate(location):
        location_log = location_log.append([{"ID":i, "X":coordinate[0], "Y":coordinate[1], "Camera_index":coordinate[2], "Confidence":coordinate[3]}])

print("type:{}".format(location_log.shape))
#location_log.to_csv("test.csv")
print("updated_location is:{}".format(location_log))
t1 = (124, 457, 2)
t2 = (465, 872, 67)
b.append(t1)
b.append(t2)
print(b)
print(b[0])
for i, item in enumerate(b):
    print("content {} is {}".format(i, item))
    for j in range(len(item)):
       print("index {} content in item {}".format(j,item[j]))