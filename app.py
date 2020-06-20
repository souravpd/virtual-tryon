import streamlit as st
from PIL import Image
from script import predict
import time

st.title("Virtual Try ON")

cloth1 = Image.open('./data/cloth/002337_1.jpg')
cloth2 = Image.open('./data/cloth/002599_1.jpg')
cloth3 = Image.open('./data/cloth/003086_1.jpg')

st.sidebar.image(cloth1, caption="002337", width=100, use_column_width=False)
st.sidebar.image(cloth2, caption="002599", width=100, use_column_width=False)
st.sidebar.image(cloth3, caption="003086", width=100, use_column_width=False)

uploaded_person = st.file_uploader("Upload a Photo", type="jpg")
user_input = st.text_input("Enter the File Name")
selected = st.selectbox('Select the Item Id:', [
                        '', '002337', '002599', '003086'], format_func=lambda x: 'Select an option' if x == '' else x)


if uploaded_person is not None:
    person = Image.open(uploaded_person)
    st.image(person, caption=user_input, width=100, use_column_width=False)

if st.button('Execute'):
    f = open("./Database/val_pairs.txt" , "w")    
    f.write(user_input+" "+selected+"_1.jpg")
    f.close()
    predict()
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.08)
        progress_bar.progress(percent_complete + 1)
    result = Image.open("./output/second/TOM/val/" + selected + "_1.jpg")
    st.image(result , caption="Result" , width=500 , use_column_width=False)



