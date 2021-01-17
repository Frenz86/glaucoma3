import streamlit as st
#import plotly.figure_factory as ff
from PIL import Image
import pandas as pd
import numpy as np

def show_footer():
    st.markdown("***")
    st.markdown("**Like this tool?** Follow us on "
                "[Linkedin](https://twitter.com/xxxxxxx).")

def main():
    st.button("Re-run")
    # set up layout
    st.title("How to crop correctly the image")
    image = Image.open('cropped.jpg')
    st.image(image, caption='',use_column_width=True)


    #st.markdown("Coming soon ... Sign up [here]() to get notified.")
    show_footer()

if __name__ == "__main__":
    main()


