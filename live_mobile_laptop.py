import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


model_path = "mobile_laptop_classifier0.916.keras"
model = load_model(model_path)

def live_object_detection(model):
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        if not ret:
            print("faild to connect camera")
            break
        
        img = cv2.resize(frame,(200,200))
        img_to_array = tf.keras.preprocessing.image.img_to_array(img)
        img_to_array = np.expand_dims(img_to_array,axis=0)
        img_to_array /= 255
        
        predi = model.predict(img_to_array)
        print(f"Prediction: {predi}")
        
        if predi > 0.5:
            cv2.putText(frame,"Mobile",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,"Laptop",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
        cv2.imshow("Live Recognition",frame)
        
        key = cv2.waitKey(1)
        if key == 27:
            break
        
    vid.release()
    cv2.destroyAllWindows()
        
live_object_detection(model)