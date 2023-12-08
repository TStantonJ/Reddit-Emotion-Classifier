import streamlit as st
from training_page import  train_side_bar, train_data_tabs, train_model_tab, train_analysis_tab
from analysis_page import analysis_model_tab
# %%--------------------------------------------------------------------------------------------------------------------
mainTab1, mainTab2= st.tabs(["Sentiment Training", "Sentiment Logging Training",])
with mainTab1:
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
    # %%--------------------------------------------------------------------------------------------------------------------
    code = """df = pd.DataFrame(
        np.random.randint(low=0, high=100, size=(10, 10)),
        columns=('col %d' % i for i in range(10)))

    st.dataframe(df)"""
    st.code(code, language="python")

    print('Amir')
with mainTab2:
    st.title("Sentiment Logging Training")
    analysis_model_tab()


