# Face Detection and Recognition
Facial Recognition or Facial Image Detection can be used to identify the individuals using their facial features.

Facial Recognition or Face Detector can be used to know whether a persons Identity through his/her face features. This Repository can be used to carry out such a task. It uses your WebCamera and then identifies your expression in Real Time.

# PLAN
Implementation of OpenCV HAAR CASCADES
I'm using the "Frontal Face Alt" Classifier for detecting the presence of Face in the WebCam. This file is included with this repository. You can find the other classifiers here.

Next, we have the task to load this file, which can be found in the label.py program. E.g.: 
We load the xml file
classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Implementation
![](Face-Detection.gif)



