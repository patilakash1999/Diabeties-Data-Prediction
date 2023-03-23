
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_option_menu import option_menu
import webbrowser
import plotly.express as exp
import plotly.io as pio
import seaborn as sns


df=pd.read_csv('Diabetes.csv')
a=df.head(5)
# def plot_data(df,varx,vary,target):
#     pio.templates.default="simple_white"
#     exp.defaults.template="ggplot2"
#     exp.defaults.color_continuous_scale=exp.colors.sequential.Blackbody
#     exp.defaults.width=1200
#     exp.defaults.width=800
#     fig=exp.scatter(df,x=varx,y=vary,color=target)
#     fig.show()    
# plot_data(df,"Glucose","Age","Outcome")  
        




loaded_model=pickle.load(open('trained_model.sav','rb'))



#Creating a function for prediction
def diabetes_Prediction(input_data):
    
  

#changing the input_data to numpy array
    input_data_as_numpy_array=np.asarray(input_data)

#reshape the array as we are predicting for one instance
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

#standarized the input data
#std_data=scaler.transform(input_data_reshaped)
#print(std_data)

    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)

    if( prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'



    

def main():
   
    with st.sidebar:
        selected=option_menu(
            menu_title="DiabetesMedicalTest", #required
            options=["Home"], #required


    )

    
    if selected=="Home":
        st.title(f"....WELCOME TO THE DIABETES DATA PREDICTION....")
        st.subheader('Diabetes Prediction Test')
        st.text('Here You Can Test Your Diabetes')
    

    

    #getting input data from user
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age  

    Pregnancies=st.text_input('Number of Pregnancies')
    Glucose=st.text_input('Glucose level')
    BloodPressure=st.text_input('Blood Pressure Value')
    SkinThickness=st.text_input('Skin Thickness value')
    Insulin=st.text_input('Insulin Level')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function value')
    Age=st.text_input('Age Of a Person')

    #code for Prediction
    diagnosis = ''

    #create button prediction

    if st.button('Diabetes Test Result'):
        diagnosis=diabetes_Prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])


    st.success(diagnosis)


if __name__ == '__main__':
    main()

with st.sidebar:
    selected=option_menu(
        menu_title="Main Menu", #required
        options=["Dataset","Graphs","Accuracy Score","Contact"],
    ) #required
if selected=="Dataset":
    st.table(a)

if selected=="Graphs":
    st.title("Graphs")
    st.bar_chart(df['Age'])


    sns.scatterplot(data=df, x="Glucose", y="BMI", hue="Age", size="Age")
    # sns.scatterplot(data=df, x="BloodPressure", y="Age", hue="BloodPressure",palette="prism")
    st.pyplot()
    sns.scatterplot(data=df, x="BloodPressure", y="Age", hue="BloodPressure",palette="prism")
    st.pyplot()
    sns.countplot('Outcome', data = df)
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)

    showPyplotGlobalUse = False
if selected=="Accuracy Score":
    st.header("Accuracy score of the training data: 0.7866449511400652")
    st.header("Accuracy score of the test data: 0.7727272727272727")

if selected=="Contact":
    if st.button("Contact with Email") :
        st.text("Akash Patil")
    if st.button("Contact with whatsapp"):
        st.text("12346747")

menu=["LogIn","SignUp"]
choice=st.sidebar.selectbox("Menu",menu)
if choice=="LogIn":
    st.subheader("LogIn")
    
    Email=st.text_input("UserName") #required
    Password=st.text_input("Password") #required
    if st.button("LogIn"):
        st.text("Sucessfully logIn")
if choice=="SignUp":
    st.subheader("SignUp")
    FirstName=st.text_input("First Name")
    LastName=st.text_input("Last Name")
    Email=st.text_input("UserName")
    Password=st.text_input("Password")
    if st.button("LogIn"):
        st.text("Sign up Sucessfully done!")
    
    # x=st.radio("Graphs",["Barchart"],0)

    # if "Graphs"=="Barchart":
        # st.barchart(df['Age'])
    



      


  