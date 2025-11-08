# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 18:59:37 2025

@author: shoun
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('G:/AI_ML/Deploy_ML/trained_model .sav','rb'))

#Creating a function for prediction
def diab_pred(input_data):
    input_data = (8,125,96,0,0,0,0.232,54)

    #Makuing the input data into a numpy array
    input_data_as_nparray = np.asarray(input_data)

    #Reshaping the array for predicting one instance
    input_data_reshaped = input_data_as_nparray.reshape(1,-1)

    #Standardize the input data
    #std_data = scaler.transform(input_data_reshaped)
    #print(std_data)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction == 0):
      return "Non-Diabetic"
    else:
      return "Diabetic"
  
def main():
    st.title("Diabetes Prediction Web App")
    
    #getting input data from the user
    # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    Preg = st.text_input("Number of preganencies: ")
    glc = st.text_input("Blood Glucose Level: ")
    BldPres = st.text_input("Blood Pressure Value: ")
    SKthk = st.text_input("Skin Thickness: ")
    Insul = st.text_input("Blood Insulin level: ")
    BMI = st.text_input("BMI value: ")
    DiabPed = st.text_input("Diabetes Pedigree function value: ")
    Age = st.text_input("Age: ")
    
    #Prediction code
    diagnosis = ''
    
    if st.button("Diabetes Test Result"):
        diagnosis = diab_pred([Preg, glc, BldPres, SKthk, Insul, BMI, DiabPed, Age])
        
    st.success(diagnosis)
    
if __name__ == '__main__':  #We did this so that the file will only execute when run from cmd
    main()