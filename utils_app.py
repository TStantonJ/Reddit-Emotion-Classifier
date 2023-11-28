
import  streamlit as st
import os

def side_bar():
    with st.sidebar:
        if st.button('Reset'):
            # Clear the chat history
            st.session_state.messages = []

def tabs():
    tab1, tab2, tab3 = st.tabs(["🗃 Data", " Model", " Analysis"])
    with tab1:
        st.subheader("A tab with the data")
        st.button("Data Select", type="primary")

        uploaded_file = st.file_uploader("Choose a file")
        print(uploaded_file)
        if uploaded_file is not None:
            st.write("You selected the file:", uploaded_file.name)
            st.session_state.data = uploaded_file

        st.session_state.data = ""
        if st.button('Say hello'):
            filenames = os.listdir('.')
            selected_filename = st.selectbox('Select a file', filenames)
            file_path = os.path.join('.', selected_filename)
            st.session_state.data = file_path
            st.write('Goodbye')
        else:
            st.write('Goodbye')
        
    with tab2:
        st.subheader("A tab with the data")
    with tab3:
        st.subheader("A tab with the results")

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)
