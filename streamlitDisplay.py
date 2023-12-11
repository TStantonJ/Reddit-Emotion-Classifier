import streamlit as st
from training_page import  train_data_tabs, train_model_tab, train_analysis_tab
from analysis_page import analysis_data_tabs, analysis_model_tab
<<<<<<< Updated upstream:streamlitDisplay.py
# %%--------------------------------------------------------------------------------------------------------------------
mainTab1, mainTab2= st.tabs([ "Sentiment Logging Training", "Sentiment Training",])
=======

"""
Main function calls of the project.
mainTab1: Core functionality of the program, allows for applying models to live data.
mainTab2: Allows for demonstraition of models and training of new bert models.
"""
mainTab1, mainTab2= st.tabs([ "Sentiment Logging", "Model Training & Analysis",])

# Main analysis/application page
>>>>>>> Stashed changes:streamlit_display.py
with mainTab1:
    st.title("Sentiment Logging Training")
    analysis_data_tabs()
    analysis_model_tab()

# Model verification/training page
with mainTab2:
<<<<<<< Updated upstream:streamlitDisplay.py
    st.title("Sentiment Training of models")
    # %%--------------------------------------------------------------------------------------------------------------------
    # Session variables and utility function instantiation
=======
    st.title("Model Training & Analysis")
>>>>>>> Stashed changes:streamlit_display.py
    st.session_state.dataSource = "None Selected"

    # Data Section
    train_data_tabs()
    st.divider()

    # Model Section
    train_model_tab()
    st.divider()

<<<<<<< Updated upstream:streamlitDisplay.py
    # Analysis Section
    train_analysis_tab()



# %%
=======
>>>>>>> Stashed changes:streamlit_display.py
