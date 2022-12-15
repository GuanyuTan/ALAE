import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from PIL import Image
import os
import tkinter as tk
from tkinter import filedialog
import time



st.title("Dashboard")
image_path = ["/home/jaws/data/fyp_covid/data_1/test"]
classes = [0]
root, dir, files= next(os.walk(image_path[0],topdown=False))
image_list = []
for file in files[:3]:
    image = Image.open(root+"/"+file)
    image_list.append(image)
with st.beta_container():
    st.title("COVID+")
    st.image(image_list)
with st.beta_container():
    st.title("COVID-")
    st.image(image_list)

root = tk.Tk()
root.withdraw()

# Make folder picker dialog appear on top of other windows
root.wm_attributes('-topmost', 1)
clicked = st.sidebar.button('Dataset Picker')
if clicked:
    dirname = st.sidebar.text_input('Selected folder:', filedialog.askdirectory(master=root))


option = st.sidebar.selectbox(
    "Resolution",("128","256","512")
)
st.sidebar.write("Inference Progress")
my_bar = st.sidebar.progress(0)

for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1)

score = 200
performance = st.sidebar.write(f"Performance:{score} IS")

button = st.sidebar.button("Save",key="something")
if button:
    st.write("Saved!")