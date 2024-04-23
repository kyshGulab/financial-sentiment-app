import streamlit as st
import joblib

model = joblib.load('model.joblib')

def predict_text(model, text):
    prediction = model.predict([text])
    prediction_proba = model.predict_proba([text])
    class_labels = model.classes_
    proba_scores = {class_labels[i]: prediction_proba[0][i] for i in range(len(class_labels))}
    return prediction[0], proba_scores


st.title('Text Classification App')
user_input = st.text_area("Enter financial news to see if its positive or negative:")
if st.button('Predict'):
    prediction, scores = predict_text(model, user_input)
    st.write("This news is:", prediction)
