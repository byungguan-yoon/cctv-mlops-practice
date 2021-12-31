import streamlit as st

import numpy as np
import torch
from PIL import Image

from ast import literal_eval

import requests

from cnn_model import FaceModel
from prepro import preprocess

import sys
import os
sys.path.append(os.path.join(os.getcwd(), '../Utils'))
ImageEncoder = __import__("ImageEncoder")

st.set_page_config(layout="wide")

@st.cache
def load_emp_feat() -> dict:
    return np.load('./emp_emb_feat.npy')


def main():
    if 'model' not in st.session_state:
        st.session_state.model = FaceModel()
    if 'emp_feat' not in st.session_state:
        st.session_state.emp_feat = load_emp_feat()
    
    image_uploaded = st.sidebar.file_uploader("Image Upload:", type=["png", "jpg", "jpeg"])
    if image_uploaded:
        image_origin = Image.open(image_uploaded)
        image_origin = np.array(image_origin.convert('RGB'))
        image_bytes = ImageEncoder.Encode(image_origin, ext='jpg', quality=90)
        response = requests.post('http://0.0.0.0:8786/inference', files={'image': image_bytes})
        max_sim_score = response.content.decode()
        max_sim_score = literal_eval(max_sim_score)['max_sim_score']
        max_sim_score = float(max_sim_score)
        st.image(image=image_origin, caption=f'similar score : {max_sim_score}')

        # st.image(image=image_origin, caption=f'similar score : {max_sim_score}')
        if max_sim_score >= 0.98:
            st.balloons()

main()