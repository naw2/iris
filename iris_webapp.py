import streamlit as st
import joblib
import numpy as np
st.set_page_config(page_title = "My Iris Test")
st.title("Iris Flower Classification")
@st.cache(allow_output_mutation = True)
def get_model():
    return joblib.load('iris_knn_model.joblib')
spl = st.text_input("Enter Sepal Length:", "")
spw = st.text_input("Enter Sepal Width:", "")
pel = st.text_input("Enter Petal Length:", "")
pew = st.text_input("Enter Petal Width:", "")
if st.button("Classify flower"):
    values = [spl,spw,pel,pew]
    num_values = []
    for x in  values:
        num_values.append(float(x))
    num_values = np.asarray(num_values).reshape(1,-1)
    model = get_model()
    pred = model.predict(num_values)
    pred= int(pred)
    if pred == 0:
        st.write("Flower is Iris_Sentosa")
    elif pred == 1:
        st.write("Flower is Vesina")
    else:
        st.write("Flower is Versicolor")