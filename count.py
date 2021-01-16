from dop.centroidtracker import CentroidTracker
from dop.trackableobject import TrackableObject
from imutils.video import FPS
import numpy as np
import imutils
import time
import dlib
import cv2

# лист классов для MobileNet SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
# Загрузка модели
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

vs = cv2.VideoCapture("video.avi")

# размер кадра(опредилиться дальше)
W = None
H = None
# создаем трекер центроидов, список трекеров dlib, словарь для сопастовления objectID и TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=100)
trackers = []
trackableObjects = {}
# инициализация счетчиков
totalFrames = 0
down = 0
up = 0
# счетчик fps
#fps = FPS().start()
skip = 30
conf = 0.4

print("Старт обработки. Ждите")
while True:
    frame = vs.read()
    frame = frame[1]

    # если frame = None то запись завершилась и заканчиваем цикл
    if frame is None:
        break

    # уменьшаем размер кадра, для ускорения процесса
    frame = imutils.resize(frame, width=500)
    # конвертируем в RGB для dlib
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    #status = "Waiting"
    rects = []
    # Каждые "skip" кадров детектим новые обьекты
    if totalFrames % skip == 0:
        # инициализируем трекеры
        # status = "Detecting"
        trackers = []
        # Находим все обьекты
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # проходим по всем обьектам
        for i in np.arange(0, detections.shape[2]):
            # вытаскиваем вероятность предсказания
            confidence = detections[0, 0, i, 2]
            # убираем слабые предсказания меньше чем "conf"
            if confidence > conf:
                # номер класса
                idx = int(detections[0, 0, i, 1])
                # если этот класс не человек то игнорируем его
                if CLASSES[idx] != "person":
                    continue
                # вычисляем ограниччивающий прямоугольник
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                # создаем новый трекер и начинаем "следить"
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)
                # добавляем в общий список трекеров
                trackers.append(tracker)

    # отслеживаем
    else:
        # перебираем все трекеры
        for tracker in trackers:
            # статус трекинга
            # status = "Tracking"
            # узнаем и берем новую позицую
            tracker.update(rgb)
            pos = tracker.get_position()

            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            rects.append((startX, startY, endX, endY))

    # рисуем линию через которую проходят люди
    #cv2.line(frame, (0, 2 * H // 3), (W, 2 * H // 3), (0, 0, 255), 2)
    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids
    # используем трекер центроидов для сопоставления старых и новых(вычисленных) центроидов
    objects = ct.update(rects)

    # перебираем отслеживаемые обьекты
    for (objectID, centroid) in objects.items():
        # проверка на существование отслеживаемого обьекта
        to = trackableObjects.get(objectID, None)
        # если нет, то создаем
        if to is None:
            to = TrackableObject(objectID, centroid)
        # выясняем в какую сторону движется человек
        else:
            # если разница Y координат текущего и предыдущих направлений отрицательна, то движется вверх
            # если разница Y координат текущего и предыдущих направлений положительна, то движется вниз
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
            # проверяем считали мы этот обьект
            if not to.counted:
                # если разница отрицательна и находится выше линии то движется наверх
                if direction < 0 and centroid[1] < 2 * H // 3:
                    up += 1
                    to.counted = True
                # если разница положительна и находится ниже линии то движется наверх
                elif direction > 0 and centroid[1] > 2 * H // 3:
                    down += 1
                    to.counted = True
        # сохранчем трекер обьекта для обработки в следующих кадрах
        trackableObjects[objectID] = to

        # отображение центроидов и другой информации на кадре
        # text = "ID {}".format(objectID)
        # cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    # info = [
    #     ("Up", totalUp),
    #     ("Down", totalDown),
    #     ("Status", status),
    # ]
    # for (i, (k, v)) in enumerate(info):
    #     text = "{}: {}".format(k, v)
    #     cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    #cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # q для выхода из цикла
    if key == ord("q"):
        break
    #обновляем fps
    totalFrames += 1
    #fps.update()
# останавливаем счетчик fps
#fps.stop()
# print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("Down ", down)
print("Up ", up)

vs.release()
cv2.destroyAllWindows()