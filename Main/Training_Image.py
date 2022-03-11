import os
import time
import cv2
import numpy as np
from PIL import Image
from threading import Thread

#-------------------image label-------------------------
def get_images_and_labels(path):
    #get the path of all the files in the folder
    image_paths = [os.path.join(path,f) for f in os.listdir(path)]
    #create face list
    faces = []
    #create id list
    ids = []
    #looping through all the image paths and loading the ids and the images
    for image_path in image_paths:
        #loading the image and converting it to gray scale
        pil_image = Image.open(image_path).convert('L')
        #converting the PIL image into numpy array
        image_np = np.array(pil_image, 'uint8')
        #getting the id from image
        Id = int(os.path.split(image_path)[-1].split(".")[1])
        #extract the face from the traning image sample
        faces.append(image_np)
        ids.append(Id)
        return faces, ids


#----------------training images function--------------------
def train_images():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    faces, ids = get_images_and_labels("./Training_Image/")
    Thread(target= recognizer.train(faces, np.array(ids))).start()
    #optional for a visual counter effect
    Thread(target= counter_img("./Data/Training_Image/")).start()
    recognizer.save("./Data/Training_Image/"+os.sep+"Trainner.yml")
    print("Finished")



# Optional, adds a counter for images trained (You can remove it)
def counter_img(path):
    img_counter = 1
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in image_paths:
        print(str(img_counter) + " Images Trained", end="\r")
        time.sleep(0.008)
        img_counter += 1
        
train_images()

