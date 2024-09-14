import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st

@st.cache_resource
def load_my_model():
    model_path = "mobile_laptop_model/200200_mobile_laptop_classifier0.916.keras"
    return load_model(model_path)

model = load_my_model()

if 'run' not in st.session_state:
    st.session_state['run'] = False

def live_object_detection(model):
    
    frame_holder = st.empty()
    if st.button("start"):
        st.session_state['run'] = True
    if st.button("Stop"):
        st.session_state['run'] = False
    
    vid = cv2.VideoCapture(0)
    while st.session_state['run']:
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
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_holder.image(frame, channels="RGB")    
        
        if not st.session_state['run']:
            break
        
    vid.release()
    cv2.destroyAllWindows()
    
        
live_object_detection(model)