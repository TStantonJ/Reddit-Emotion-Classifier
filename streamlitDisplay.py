import streamlit as st
from training_page import  train_side_bar, train_data_tabs, train_model_tab, train_analysis_tab
from analysis_page import analysis_data_tabs, analysis_model_tab
# %%--------------------------------------------------------------------------------------------------------------------
mainTab1, mainTab2= st.tabs([ "Sentiment Logging Training", "Sentiment Training",])
with mainTab1:
    st.title("Sentiment Logging Training")
    analysis_data_tabs()
    analysis_model_tab()


with mainTab2:
    st.title("Sentiment Training of models")
    # %%--------------------------------------------------------------------------------------------------------------------
    # Session variables and utility function instantiation
    st.session_state.dataSource = "None Selected"
    train_side_bar()

    # Data Section
    train_data_tabs()
    st.divider()

    # Model Section
    train_model_tab()
    st.divider()

    # Analysis Section
    train_analysis_tab()



# %%
