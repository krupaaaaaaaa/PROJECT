import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import cv2


model = load_model('Skin_disease1.keras')

def Image_detection(img):
    img_height , img_width = 160,160
    image = tf.keras.utils.load_img(img , target_size = (img_height, img_width))
    img_arr = tf.keras.utils.array_to_img(image)
    img_bat = tf.expand_dims(img_arr,0)
    data_cat=[
        'Actinic keratosis',
        'Atopic Dermatitis',
        'Benign keratosis',
        'Dermatofibroma',
        'Melanocytic nevus',
        'Melanoma',
        'Squamous cell carcinoma',
        'Tinea Ringworm Candidiasis',
        'Vascular lesion',
        'normal skin'
    ]

    predict = model.predict(img_bat)

    score = tf.nn.softmax(predict)
    st.image(img , width=200)
    st.write('Predicted class is ' + data_cat[np.argmax(score)])
    st.write('Predicted with an accuracy of ' + str(np.max(score)*100))
    return data_cat[np.argmax(score)]



def capture_image():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Scanner ")
    img_counter=0
    while True:
        ret, frame = cam.read()
    
        if not ret:
            st.write("failed to scan")
            break
        
        cv2.imshow("test",frame)
    
        k=cv2.waitKey(1)
    
        if k%256==27:
            st.write("closing scanner")
            break
        elif k%256==32:
            img_name="image{}.png".format(img_counter)
            cv2.imwrite(img_name,frame)
            st.write("scanned")
            img_counter+=1
        

    cam.release()
    cv2.destroyAllWindows()
#img_list=[]
    for i in range(img_counter):
       value = Image_detection('image{}.png'.format(i))
    return value

def display_content(value):
    if value == 'Actinic keratosis':
        file = open('actinic_keratosis.txt','r')
        lines = file.read()
        st.write(lines)
        
    

            

with st.sidebar:
    selected = option_menu(None, ["Home", 'Disease Detection'], 
            icons=['house', 'skin'],  default_index=0, )
    styles = {
            "container":{"padding": "5!important","nav-link":{"color":"white","font-size":"20px"}}
        }

if selected == 'Home':
    st.header("Skin Disease Detection Model")
    st.write("Skin diseases encompass a wide range of conditions affecting the skin, which is the body's largest organ. ")
    st.write("Symptoms of skin diseases can include **itching, redness, swelling, pain, rash, blisters, and changes in skin color or texture**. Treatment varies depending on the specific condition and may include topical medications, oral medications, lifestyle changes, or procedures like surgery or light therapy. Early diagnosis and treatment are crucial to managing skin diseases effectively and preventing complications.")
    st.write("You can use this model to detect what type of skin disease you have.")
elif selected == 'Disease Detection':
    options = option_menu(None, ["Upload Image", 'Capture Image'], 
            icons=['cloud','camera'], default_index=1, orientation="horizontal")
    if options == 'Upload Image':
        image = st.file_uploader(" ",type=['jpg','png','jpeg'])
        if st.button('Get Result'):
            value = Image_detection(image)
            #if(st.button('Get remedies',on_click=display_content(value))):
                #st.write(".")
    elif options == 'Capture Image':
        st.write("Press **spacebar** to capture image and **esc** to close camera")
        value = capture_image()



    
    
