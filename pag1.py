import streamlit as st
from tflite_runtime.interpreter import Interpreter 
from PIL import Image, ImageOps
import numpy as np
import requests
import os
from io import BytesIO
import wget


def download_model():
    model_path = 'my_model2.tflite'
    if not os.path.exists(model_path):
        url = 'https://frenzy86.s3.eu-west-2.amazonaws.com/python/models/my_model2.tflite'
        filename = wget.download(url)
    else:
        print("Model is here.")

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file inside images collections: ', filenames)
    return os.path.join(folder_path, selected_filename)

def main():
    #st.title("Eye Detection")
    image_file = st.file_uploader("Upload Image", type = ['jpg','png','jpeg'])
    download_model()
    model_path = 'my_model2.tflite'

    if image_file != None:
        image1 = Image.open(image_file)
        rgb_im = image1.convert('RGB') 
        image = rgb_im.save("saved_image.jpg")
        image_path = "saved_image.jpg"
        st.image(image1, width = 450)

    else:
        folder_path = './images/'
        filename = file_selector(folder_path=folder_path)
        st.write('You selected `%s`' % filename)
        image = Image.open(filename)
        image_path = filename
        print(image_path)
        st.image(image,width = 450)
        #st.image(image,use_column_width=True)

    if st.button("Make Prediction"):
        img = Image.open(image_path)
        ## Load model
        interpreter = Interpreter(model_path)
        print("Model Loaded Successfully.")
        ## Prepare the image
        #img = Image.open("img/test.jpg")
        image = ImageOps.fit(img, (100,100),Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)
        input_data = image[np.newaxis,...]

        ## run inference
        interpreter.allocate_tensors()
        inputdets = interpreter.get_input_details()
        outputdets = interpreter.get_output_details()
        interpreter.set_tensor(inputdets[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(outputdets[0]['index']) 
        pred = prediction[0][0]
        print(pred)     
        if(pred > 0.5):
            st.write("""
                     ## **Prediction:** You eye is Healthy. Great!!
                     """
                     )
        else:
            st.write("""
                     ## **Prediction:** You have an high probability to be affected by Glaucoma. Please consult an ophthalmologist as soon as possible.
                     """
                     )
if __name__ == '__main__':
    main()
