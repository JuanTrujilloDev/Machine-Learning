# -*- coding: utf-8 -*-
"""
Created on Sat May 23 16:01:17 2020

@author: sarsu
"""


import cv2
import numpy as np


#Importing HAAR CASCADE CLASSIFIER!

haar = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')

#Creating functions to extract images of each frame of the video

def face_ext(img):
    
    #img, scale factor = 1.3, min neighbours = 5
    faces = haar.detectMultiScale(img, 1.3, 5)
    
   
    if faces is ():
        return None
    
    #Cropping al faces 
    # X,Y coordinates Width and Height
    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        #Cropped face frame is between y,y+height+50 and x,x+width+50
        cropped_face = img[y:y+h+50, x:x+w+50]
        
    return cropped_face

#start recording

cap = cv2.VideoCapture(0)
count = 0

#Collecting 400 samples of each person

while True:
    
    ret, frame = cap.read()
    #If a face is detected
    if face_ext(frame) is not None:
        count += 1
        #Each image is resized to 400x400
        face = cv2.resize(face_ext(frame), (400, 400))
        
        #Saving file in the path
        file_name_path = 'Images/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)
        
        #Putting count on images and displaying live count
        
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        print("Face not found")    
        pass
    
    if cv2.waitKey(1) & 0xFF == ord('q') or count == 400: 
        break

cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete!")
        
        