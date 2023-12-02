
import  streamlit as st
import os
import pandas as pd

def side_bar():
    with st.sidebar:
        if st.button('Reset'):
            # Clear the chat history
            st.session_state.messages = []

def tabs():
    tab1, tab2, tab3 = st.tabs(["🗃 Data", " Model", " Analysis"])
    with tab1:
        data_tab()
        
    with tab2:
        st.subheader("A tab with the data")
    with tab3:
        st.subheader("A tab with the results")

def data_tab():
    st.subheader("Data Sources")
    
    # Toggle to select whether preloaded data is used or data is gotten life
    dataMode_toggle = st.toggle('Pre-Loaded Data mode')
    if dataMode_toggle:
        
        # Options for preloaded data
        dataSource = pd.DataFrame()
        dataOption_selectbox = st.selectbox(
        'Select Ddta from Pre-Loaded sources',
        ('Reddit Source A', 'Reddit Source B', 'Reddit Source C'),
        index=None,
        placeholder="Select data source...",)

        if dataOption_selectbox == 'Reddit Source A':
            st.write('Set')
            dataSource = pd.read_csv('preloadedData/merged_reddit_data.csv')
            st.session_state.dataSource = dataSource
        elif dataOption_selectbox == 'Reddit Source B':
            pass
        elif dataOption_selectbox == 'Reddit Source C':
            pass

        st.dataframe(dataSource, use_container_width=True)
    else: 
        st.write('Select Supported SubReddit')

    container_2 = st.empty()
    button_A = container_2.button('Btn A')
    if button_A:
        container_2.empty()
        button_B = container_2.button('Btn B')
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

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)
