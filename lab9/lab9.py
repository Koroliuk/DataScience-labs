import os
import cv2
import numpy as np

modelFile = "./model/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "./model/deploy.prototxt.txt"
inputDir = "faces"
outputDir = "result"

net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

results = []
images = os.listdir(inputDir)
for image in images:
    full_path = os.path.join(inputDir, image)
    img = cv2.imread(full_path)
    img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 117.0, 123.0))

    net.setInput(blob)
    faces = net.forward()
    result = img.copy()
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            box = faces[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(result, (x, y), (x1, y1), (255, 0, 0), 2)

    results.append(result)

for i in range(len(results)):
    cv2.imshow("img", results[i])
    cv2.imwrite(f"./result/{i + 1}.jpg", results[i])
    cv2.waitKey()
