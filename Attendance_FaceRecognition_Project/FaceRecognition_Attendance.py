"""
Created on Sat May 15 14:25:32 2021

@author: Anwarrior

Added Features by Anwarrior: 
    1.) End when All Present
    2.) Voice Output 
    3.) If name is already in Present list Voice Output Asks For next person
"""

import cv2
import numpy as np
import pandas as pd
import face_recognition
import os
from datetime import datetime
import pyttsx3
import sys

#from PIL import ImageGrab
 
path = 'C:/Users/Admin/Documents/Face rec'          #location to images of Student
images = []
classNames = []
myList = os.listdir(path)

print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 
def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
 
 
encodeListKnown = findEncodings(images)
print('Encoding Complete')
 
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
 

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    
    Atdf = pd.read_csv("D:/Desktop/Face_Attendance_Enhanced_Project/Attendance.csv")  #Location for attendance.csv file
    Atlist = Atdf.iloc[:,0].values.tolist()
    converter = pyttsx3.init()
    
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        name = classNames[matchIndex].upper()
        
        if (len(Atlist) != len(classNames)):
            if name in Atlist:
                converter.say("Next Person Please") 
                converter.runAndWait()
            else:
                converter.say(name)   
                converter.runAndWait()
            
        if matches[matchIndex]:
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
            
        
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
    
    if len(Atlist) == len(classNames):
        cap.release()
        cv2.destroyAllWindows()
        converter.say("All Present")   
        converter.runAndWait()
        sys.exit()   