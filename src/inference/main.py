import streamlit as st

import numpy as np
import torch
from PIL import Image

from cnn_model import FaceModel
from prepro import preprocess

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

        img_tensor = preprocess(image_origin)
        img_feat = st.session_state.model(torch.unsqueeze(img_tensor, dim=0))
        img_feat_np = img_feat.detach().numpy()

        sim_score = np.dot(st.session_state.emp_feat, img_feat_np.T)
        argmax_sim_score = np.argmax(sim_score)
        max_sim_score = sim_score.max()
        st.image(image=image_origin, caption=f'similar score : {max_sim_score}')
        if max_sim_score >= 0.98:
            st.balloons()

main()