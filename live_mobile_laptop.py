import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),  # Save logs to a file
        logging.StreamHandler()          # Output logs to the terminal
    ]
)
# Cache the model so it's only loaded once else error in model server
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
            logging.info("faild to connect camera")
            break
        
        img = cv2.resize(frame,(200,200))
        img_to_array = tf.keras.preprocessing.image.img_to_array(img)
        img_to_array = np.expand_dims(img_to_array,axis=0)
        img_to_array /= 255
        
        predi = model.predict(img_to_array)
        logging.info(f"Prediction: {predi}")
        
        if predi > 0.5:
            cv2.putText(frame,"Mobile",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            logging.info(f"Prediction: Mobile")
        else:
            cv2.putText(frame,"Laptop",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            logging.info(f"Prediction: Laptop")
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_holder.image(frame, channels="RGB")    
        
        cv2.waitKey(100) # to set frame per second (delay)
        if not st.session_state['run']:
            break
        
    vid.release()
    cv2.destroyAllWindows()
    
        
live_object_detection(model)