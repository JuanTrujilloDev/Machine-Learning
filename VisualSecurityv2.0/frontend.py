# -*- coding: utf-8 -*-
"""
Created on Sat May 23 16:58:03 2020

@author: sarsu
"""

import os
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np

from keras.preprocessing import image


people_dir = 'Datasets/Test'
name_list = []
for person_name in os.listdir(people_dir):
    name_list.append(person_name)
   
print(name_list)

model = load_model('facefeatures_new_model.h5')

# Loading the cascades
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face is detected, it returns the input image
    
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Doing some Face Recognition with the webcam 
video_capture = cv2.VideoCapture(0)

while True:
   
    _, frame = video_capture.read()
    #canvas = detect(gray, frame)
    #image, face =face_detector(frame)
    
    face=face_extractor(frame)
    
    if type(face) is np.ndarray:
        face = cv2.resize(face, (244, 244))
        im = Image.fromarray(face, 'RGB')
           #Resizing into 224x244 because we trained the model with this exact image size.
        img_array = np.array(im)
                    #Our keras model used a 4D tensor with 4 parameters, (images x height x width x channels)
                    #So we must change the dimension from 128x128x3 into 1x128x128x3 
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
       
                     
        
        value = 0
        
        #IF the accuracy of the prediction is higher than 90% we print the person matched name
        if(pred[0][1]>0.9):
                value = (pred[0][1] * 100)
                value = int(value)
                cv2.putText(frame,name_list[1] + " " + str(value) + "%", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                print(name_list[1] + str(1))
                
            #if the person isn't in the trainned dataset it will print "None Matching"
        else:
            cv2.putText(frame,"None Matching", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            #If no face is found it will print "No face found"
    else:
        cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
    cv2.imshow('Video', frame)
    #If we press q it will exit the program.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()