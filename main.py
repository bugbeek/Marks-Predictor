import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


#function to predict marks
def marks_predict(hours):
    df = pd.read_csv('score.csv')

    #Feature selection
    X = df[['Hours']]
    y = df[['Scores']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)


    #initiate the model
    model = LinearRegression()

    #Fit the model
    model.fit(X_train, y_train)

    #prediction
    predicted_score = model.predict([[hours]])
    predicted_score = predicted_score[0][0].round(2)

    return predicted_score



#Input part of the code
_, col2, _ = st.columns([1, 2, 1])

with col2:
    st.markdown("<h2 style='text-align: left;'>📘 Marks Predictor</h2>", unsafe_allow_html=True)


st.text("")
st.markdown("<div style='text-align: center; font-size: 18px;'>📚 We will help you predict the marks you will get in your exams based on the number of hours you studied! ⏳</div>", unsafe_allow_html=True)
st.text("")
hours = st.slider('🎓 Please select the number of hours you studied:', min_value=0.0, max_value=13.0, step=0.1)




if hours > 0:
    predicted_score = marks_predict(hours)
    st.text("")
    st.markdown(f"<div style='text-align: center; font-size: 24px; color: #4CAF50;'><strong>🌟 Based on your study hours, our model predicts you'll score {predicted_score} marks! 🎉</strong></div>", unsafe_allow_html=True)

