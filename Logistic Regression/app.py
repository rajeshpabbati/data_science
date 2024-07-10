#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from pickle import load

st.title("Model Deployment: Logistic Regression")

st.sidebar.header("User Input Parameters")

def user_input_features():
    Pclass = st.sidebar.selectbox("Passenger Class", ("1","2","3"))
    Age = st.sidebar.number_input("Insert the Age")
    SibSp = st.sidebar.selectbox("Sibling and Spouse", ("0","1","2","3","4","5","6","7","8"))
    Parch = st.sidebar.selectbox("Parent and Child", ("0","1","2","3","4","5","6"))
    Fare = st.sidebar.number_input("Insert the Fare")
    Sex_male = st.sidebar.selectbox("Gender", ("1","0"))
    Embarked_Q = st.sidebar.selectbox("Embarked_Q", ("1","0"))
    Embarked_S = st.sidebar.selectbox("Embarked_S", ("1","0"))
    data = {"Pclass":Pclass,
            "Age":Age,
            "SibSp":SibSp,
            "Parch":Parch,
            "Fare":Fare,
            "Sex_male":Sex_male,
            "Embarked_Q":Embarked_Q,
            "Embarked_S":Embarked_S}
    features = pd.DataFrame(data,index=[0])
    return features

df = user_input_features()
st.subheader("User Input parameters")
st.write(df)

# load the model from disk
loaded_model = load(open("Logistic_Regression_Model.sav", "rb"))

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

st.subheader("Predicted Result")
st.write("Yes" if prediction_proba[0][1] > 0.5 else "NO")

st.subheader("Prediction Probability")
st.write(prediction_proba)


# In[ ]:




