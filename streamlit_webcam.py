import streamlit as st
import numpy
import cv2

st.title("hello")
frame_holder = st.empty()
stop_btn = st.button("Stop")

video = cv2.VideoCapture(0)

while video.isOpened and not stop_btn:
    ret, frame = video.read()
    if not ret:
            print("faild to connect camera")
            break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_holder.image(frame, channels="RGB")

    key = cv2.waitKey(1)
    if key == 27:
        break
        
video.release()
cv2.destroyAllWindows()