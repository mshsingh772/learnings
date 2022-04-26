import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2, os, random
import numpy as np
import pygame
from time import sleep



model = load_model("./model")
label_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

def play_music(detected_emotion):
    music_library = os.path.join("./songs",detected_emotion)
    if detected_emotion not in ["none"]:
        print("searching in", music_library)
        all_mp3 = os.listdir(music_library)

        random_song = os.path.join(music_library,random.choice(all_mp3))
        print("now playing",random_song)
        pygame.mixer.init()
        pygame.mixer.music.load(random_song)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            sleep(1)
        return True
    else:
        return False

def detect_faces_image():
    images_directory = "./images"
    emotion_images  =  os.listdir(images_directory)
    random_image = random.choice(emotion_images)
    image_path = os.path.join(images_directory,random_image)
    print("The chosen one",image_path)
    # Detecting Face in image using harcascade
    orig_image = cv2.imread(image_path)
    image = orig_image.copy()
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    num_faces = face_detector.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
    detected_emotion = "none"
    for(x,y,w,h) in num_faces:     #supporting only emotion of last detected face
        cv2.rectangle(image,(x,y-50),(x+w,y+h+10),(0,255,0),4)
        roi_gray_image = gray_image[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_image,(48,48)),-1),0)
        emotion_prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        print("emotion",label_dict[maxindex])
        detected_emotion = label_dict[maxindex]
        cv2.putText(image,label_dict[maxindex],(x+5,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
    music = play_music(detected_emotion)
    if not music:
        print("Emotion Not Supported")
    cv2.imwrite("out_image.jpg",image)
    


def detect_faces_live():
    # cap = cv2.VideoCapture(0)
    # frame_count = 0 
    cap = cv2.VideoCapture('./laugh.mp4')
    while True:
        ret,frame = cap.read()
        image = cv2.resize(frame,(1280,720))
        if not ret:
            break
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        num_faces = face_detector.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
        detected_emotion = "none"
        for(x,y,w,h) in num_faces:  #supporting only emotion of last detected face
            cv2.rectangle(image,(x,y-50),(x+w,y+h+10),(0,255,0),4)
            roi_gray_image = gray_image[y:y+h, x:x+w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_image,(48,48)),-1),0)
            emotion_prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            print("emotion",label_dict[maxindex])
            detected_emotion = label_dict[maxindex]
            cv2.putText(image,label_dict[maxindex],(x+5,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): #need to make changes here
            break
        music = play_music(detected_emotion)
        if not music:
            print("Emotion Not Supported")


        
if __name__=="__main__":
    detect_faces_image()
    # detect_faces_live()


