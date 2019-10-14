import numpy as np
import cv2
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go

print("***INFO: Loading trained model for face detection . . .")

image_path = "C:/99_Temp/Create_Data/ftest/"
result_path = "C:/99_Temp/"

f1 = "C:/99_Temp/Create_Data/f1/"
f2 = "C:/99_Temp/Create_Data/f2/"
f3 = "C:/99_Temp/Create_Data/f3/"
f4 = "C:/99_Temp/Create_Data/f4/"

outlier = "C:/99_Temp/Create_Data/outlier/"
diff_values =[]
X = []
Y = []
net = cv2.dnn.readNetFromCaffe('C:/99_Temp/deep-learning-face-detection/deploy.prototxt.txt', 'C:/99_Temp/deep-learning-face-detection/res10_300x300_ssd_iter_140000.caffemodel')

path = glob.glob(image_path + "*.jpg")
num_of_images = len(path)

for i in range(0, num_of_images, 1):

    fname = os.path.splitext(os.path.basename(path[i]))[0]
    image = cv2.imread(path[i])
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        print(len(detections))
        if confidence > 0.5 and len(detections) == 1:

            os.chdir(image_path)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10

            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            #cv2.imwrite(fname + '_.jpg', image)
            #print(startX, startY, endX, endY)
            distX = (endX - startX)
            distY = (endY - startY)

            X += [(distX)]
            Y += [(distY)]

            if distX <= 189 and distY <= 189:
                os.chdir(f1)
                cv2.imwrite(fname  + '_' + str(distX) + '_' + str(distY) + '.jpg', image)
            elif 190 >= distX <= 270 and 190 >= distY <= 270:
                os.chdir(f2)
                cv2.imwrite(fname  + '_' + str(distX) + '_' + str(distY) + '.jpg', image)
            elif 271 >= distX <= 450 and 271 >= distY <= 450:
                os.chdir(f3)
                cv2.imwrite(fname  + '_' + str(distX) + '_' + str(distY) + '.jpg', image)
            elif distX >= 451 or distY >= 451:
                os.chdir(f4)
                cv2.imwrite(fname  + '_' + str(distX) + '_' + str(distY) + '.jpg', image)
            else:
                os.chdir(outlier)
                cv2.imwrite(fname  + '_' + str(distX) + '_' + str(distY) + '.jpg', image)


            break

print("***DONE! ")
#diff_values = np.asarray(diff_values)

#np.savetxt(result_path +'xy.txt', np.c_[X,Y], delimiter = ',', fmt='%i')
#df = pd.DataFrame({"X-value" : np.asarray(X), "Y-value" : np.asarray(Y)})
#df.to_csv(result_path + "dist.csv", index = False)


trace1 = go.Histogram(
    x=X,
    marker=dict(
        color='#FFD7E9',
    ),
    opacity=0.50
)
trace2 = go.Histogram(
    x=Y,
    marker=dict(
        color='#EB89B5'
    ),
    marker_line_color='rgb(8,48,107)',
    marker_line_width=1.5,
    opacity=0.50
)

data = [trace1, trace2]
layout = go.Layout(barmode='stack')
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, validate=False, filename=result_path+'Histogram.html', auto_open=False)
