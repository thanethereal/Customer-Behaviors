from ast import For, mod
from datetime import date
import datetime
from operator import index
import os 
from tarfile import LENGTH_NAME
import time
import dlib
import cv2
from matplotlib.pyplot import flag
import pandas as pd
import mysql.connector

mydb=mysql.connector.connect(host='localhost',user='root',password='',database='customer_behaviors')
cursor=mydb.cursor()
def recognize_attendence():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("./Data/Training_Image/Trainner.yml")
    harrcascade_path = "./Model/haarcascade_frontalface_default.xml"
    facecascade = cv2.CascadeClassifier(harrcascade_path)
    df = pd.read_csv("./Report/"+os.sep+"staff_details.csv")
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns= col_names)
    
    #Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    #Define min window size to be recognized as a face
    min_w = 0.1*cam.get(3)
    min_h = 0.1*cam.get(4)
    
    while True:
        ret, image = cam.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       
        faces = facecascade.detectMultiScale(gray, 1.2, 5, minSize = (int(min_w), int(min_h)), flags = cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x,y), (x + w, y +h), (10, 159, 255), 2)
            id, conf = recognizer.predict(gray[y:y+h, x: x +w])
            if conf < 100:
                aa = df.loc[df['id'] == id]['name'].values
                confstr = " {0}%".format(round(100 - conf))
                tt = str(id) + "-" + aa
            else:
                id = 'Unknown'
                tt = str(id)
                confstr = " {0}%".format(round(100 -conf))
            if (100 -conf) > 67:
                times = time.time()
                date_stamp = date.today()
                time_stamp = datetime.datetime.fromtimestamp(times).strftime('%H:%M:%S')
                aa = str(aa)[2:-2]
                attendance.loc[len(attendance)] = [id, aa, date_stamp, time_stamp]
            tt = str(tt)[2:-2]
            if(100 -conf) > 67:
                tt = tt + "[pass]"
                cv2.putText(image, str(tt), (x +5, y -5), font, 1, (255,255,255), 2)
            if(100 -conf) <50:
                tt = "[Unknow]"
                cv2.putText(image, str(tt), (x +5, y -5), font, 1, (0,0,255), 2)
            if (100-conf) > 67:
                cv2.putText(image, str(confstr), (x + 5, y + h - 5), font,1, (0, 255, 0),1 )
            elif (100-conf) > 50:
                cv2.putText(image, str(confstr), (x + 5, y + h - 5), font, 1, (0, 255, 255), 1)
            else:
                cv2.putText(image, str(confstr), (x + 5, y + h - 5), font, 1, (0, 0, 255), 1)
        attendance = attendance.drop_duplicates(subset=['Id'], keep = 'first')
        print(attendance)
        cv2.imshow('Attendance', image)
        if (cv2.waitKey(1) == ord('q')):
            break
    times = time.time()
    date_stamp = date.today() 
    time_stamp = datetime.datetime.fromtimestamp(times).strftime('%H:%M:%S')
    Hour, Minute, Second = time_stamp.split(":")
    file_name = "./Report/"+os.sep+"Attendance.csv"
    attendance.to_csv(file_name)
    print(id)
    cursor.execute("INSERT INTO `attendance`(`DateAttendance`, `TimeAttendance`, `StaffID`) VALUES (%s,%s,%s)", (date_stamp, time_stamp, id))
    mydb.commit()
    print("Attendance Succesfull")
    cam.release()
    cv2.destroyAllWindows()

recognize_attendence()


