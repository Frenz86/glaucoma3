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

##### MAIN ####
def main():
    st.button("Re-run")
    download_model()
    model_path = 'my_model2.tflite'
    
    file = st.file_uploader("Please upload an image(jpg) file", type=["jpg"])
    if file is None:
        st.text("You haven't uploaded a jpg image file")
    else:
        img = Image.open(file)
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
