import cv2
import numpy as np
# Загрузить предопределенный словарь
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
# Сгенерировать маркер

for i in range(4):
    markerImage = np.zeros((200, 200), dtype=np.uint8)
    markerImage = cv2.aruco.drawMarker(dictionary, i, 200, markerImage, 1);
    cv2.imwrite("marker{0}.png".format(i), markerImage);

# Инициализировать параметры детектора, используя значения по умолчанию
parameters =  cv2.aruco.DetectorParameters_create()
# Обнаружение маркеров на изображении
img = cv2.imread('3.jpg')
markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)
print(markerCorners, markerIds)