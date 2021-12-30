import streamlit as st

import pandas as pd
from glob import glob
from PIL import Image
st.set_page_config(layout="wide")

img_list = glob('../data/temp_img_folder/*jpeg')


def main():
    if 'img_idx' not in st.session_state:
        st.session_state.img_idx = 0
    
    # class list
    emp_list = ['이호민', '이상규', '윤병관', '문성민']
    # file list
    img_list = glob('../data/temp_img_folder/*jpeg')
    
    img = Image.open(img_list[st.session_state.img_idx])
    
    st.image(img, width=512)

    with st.form('my_form'):
        emp_str = st.radio('employee list', emp_list)
        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            emp_idx = emp_list.index(emp_str) 
            st.write(emp_str + ' / ' + str(emp_idx))

    st.session_state.img_idx += 1
    if st.session_state.img_idx >= len(img_list):
        st.session_state.img_idx = 0


main()