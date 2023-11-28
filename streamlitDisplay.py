import streamlit as st
from utils_app import side_bar, tabs
# %%--------------------------------------------------------------------------------------------------------------------
st.title("Streamlit App Testing")
# %%--------------------------------------------------------------------------------------------------------------------
side_bar()
tabs()
st.write('This is the place you can start writing')
st.divider()
# %%--------------------------------------------------------------------------------------------------------------------
code = """df = pd.DataFrame(
    np.random.randint(low=0, high=100, size=(10, 10)),
    columns=('col %d' % i for i in range(10)))

st.dataframe(df)"""
st.code(code, language="python")

print('Amir')