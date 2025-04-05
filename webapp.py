# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 21:53:58 2025

@author: Dell
"""

import numpy as np
import pickle 
import streamlit as st

loaded_model=pickle.load(open('C:/Users/Dell/Documents/Deployement/train','rb'))

#creating a function for Prediction

def Diabetes_prediction(input_data):
    #changing the ip data to a numpy array
    input_data_as_np_array=np.asarray(input_data)
    #reshape the array as we are predicting for one instance
    input_data_reshaped=input_data_as_np_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    if(prediction[0]==0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
def main():
    
    #giving a title
    st.title('DiabetesPrediction Web App')
    
    #getting the input data from the user
    
    Pregnancies=st.text_input('Number of Pregnancies')
    Glucose=st.text_input('Glucose Level')
    BloodPressure=st.text_input('Blood Pressure value')
    SkinThickness=st.text_input('Skin Thickness Value')
    Insulin=st.text_input('Insulin Level')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input('Diabtes Pedigree fucntion value')
    Age=st.text_input('Age of the person')
    
    
    #code for Prediction
    diagnosis=''
    
    #creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis= Diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        
    st.success(diagnosis)
    
    
    
    

if __name__ == '__main__':
    main()
    
    
    
