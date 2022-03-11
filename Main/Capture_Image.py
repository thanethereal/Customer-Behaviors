import cv2
import csv
import os 
import os.path

from cv2 import COLOR_GRAY2BGR
from cv2 import imwrite
from httpx import head


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try: 
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def take_images():
    id = input("Enter your Id: ")
    name = input("Enter your Name: ")
    if (is_number(id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        haarcascadePath = "./Model/haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(haarcascadePath)
        sample_num = 0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3 , 5, minSize = (30,30), flags = cv2.CASCADE_SCALE_IMAGE)
            for (x,y,w, h) in faces:
                cv2.rectangle(img, (x,y), (x + w, y +h), (10, 159, 255), 2)
                sample_num = sample_num + 1
                cv2.imwrite("./Data/Training_Image/" + os.sep +name + "."+ id + '.' +str(sample_num)+ ".jpg", gray[y:y+h, x:x+w])
                cv2.imshow("frame", img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sample_num > 100:
                break
        cam.release()
        cv2.destroyAllWindows()
        header = ["id", "name"]
        row = [id, name]
        if (os.path.isfile("./Report/" + os.sep + "staff_details.csv")):
            with open("./Report/" + os.sep + "staff_details.csv", 'a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(i for i in header)
                writer.writerow(j for j in row)
            csvFile.close()
            
take_images()      