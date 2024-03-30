import streamlit as st
from ultralytics import YOLO
from PIL import Image 

@st.cache_resource
def mod():
    model = YOLO("best.pt")
    return model

tab1, tab2 = st.tabs(["HOME","MYMODEL"])

with tab1:
    st.title("ML Project by Umang")
    #st.title("About me")
    #st.text("My name is Umang Sharma, and I am a student at Princeton Day School, a private institution situated in the heart of Princeton. I have a passion for electrical engineering, originating from building computers. Additionally, my love for photography allows me to capture moments that tell compelling stories. I am deeply fascinated by the world of technology and constantly seek to stay updated with the latest advancements. Through my skills and expertise, I strive to make a positive impact by leveraging technology for the betterment of society."        
    #st.image("abc.jpg")

with tab2:
    st.title("Prediction")
    st.markdown("Upload an image of an animal to check it what it is")
    img = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    
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
