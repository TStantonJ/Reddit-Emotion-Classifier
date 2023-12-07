import streamlit as st
from utils_app import side_bar, data_tabs, model_tab, analysis_tab
# %%--------------------------------------------------------------------------------------------------------------------
mainTab1, mainTab2= st.tabs(["Sentiment Training", "Sentiment Logging Training",])
with mainTab1:
    st.title("Sentiment Training of models")
    # %%--------------------------------------------------------------------------------------------------------------------
    # Session variables and utility function instantiation
    st.session_state.dataSource = "None Selected"
    side_bar()

    # Data Section
    data_tabs()
    st.divider()

    # Model Section
    model_tab()
    st.divider()

    # Analysis Section
    analysis_tab()
    # %%--------------------------------------------------------------------------------------------------------------------
    code = """df = pd.DataFrame(
        np.random.randint(low=0, high=100, size=(10, 10)),
        columns=('col %d' % i for i in range(10)))

    st.dataframe(df)"""
    st.code(code, language="python")

    print('Amir')
with mainTab2:
    st.title("Sentiment Logging Training")