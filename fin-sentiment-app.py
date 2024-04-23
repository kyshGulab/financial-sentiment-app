import streamlit as st
import joblib

model = joblib.load('model.joblib')

def predict_text(model, text):
    prediction = model.predict([text])
    return prediction


st.title('Text Classification App')
user_input = st.text_area("Enter financial news to see if its positive or negative:")
if st.button('Predict'):
    prediction, scores = predict_text(model, user_input)
    st.write("This news is:", prediction)
