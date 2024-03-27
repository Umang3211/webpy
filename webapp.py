import streamlit as st
from ultralytics import YOLO
from PIL import Image 

@st.cache_resource
def mod():
    model = YOLO("best.pt")
    return model

tab1, tab2 = st.tabs(["HOME","MYMODEL"])

with tab1:
    st.title("Umang")
    st.image("abc.jpg")

with tab2:
    st.title("Prediction")
    st.markdown("U[pload Image animals to check its class")
    img = st.file_uploader("Upload an audio file", type=["jpg", "jpeg", "png"])
    
    if img is not None:
        img = Image.open(img)
        st.image(img)
        model = mod()
        res = model.predict(img)
        label = res[0].probs.top5
        conf = res[0].probs.top5conf
        conf = conf.tolist()
        col1,col2 = st.columns(2)
        col1.subheader(res[0].names[label[0]].title() +' with '+ str(conf[0])+' Confidence')
        col2.subheader(res[0].names[label[1]].title() +' with '+ str(conf[1])+' Confidence')
