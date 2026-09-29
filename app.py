import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

st.title("Passenger Survival chance in the titanic Journey")
pclass=st.slider('Enter the passenger class for the user',1,3)
sex=st.selectbox('Enter the gender of the user',['male','female'])
sibsp=st.slider('Enter the sibling spouse for the user',1,8)
parch=st.slider('Enter the passenger parent and child',1,6)
fare=st.number_input("Enter the fare of the passenger")
embarked=st.selectbox('Enter the station user started journey',['Chebourg','Queenstown','Southampton'])

data=pd.DataFrame([{'Pclass':pclass,'Sex':sex,'SibSp':sibsp,'Parch':parch,'Fare':fare,'Embarked':embarked}])

model=load_model('model.h5')

with open('labelencoder.pkl','rb') as file:
    label=pickle.load(file)

with open('onehotencoder.pkl','rb') as file1:
    onehot=pickle.load(file1)

with open('scaler.pkl','rb') as file2:
    scaler=pickle.load(file2)

data['Sex']=label.transform(data['Sex'])
embarked=onehot.transform([data['Embarked']])
embarked=pd.DataFrame(embarked,columns=onehot.get_feature_names_out())

data=pd.concat([data.drop(columns=['Embarked']),embarked],axis=1)

data[['Pclass','SibSp','Parch','Fare']]=scaler.transform(data[['Pclass','SibSp','Parch','Fare']])

y=model.predict(data)

def chance(y):
    if y>0.5:
        return("Passenger will survive the journey")
    else:
        return("Passenger will not survive the journey")

if st.button("Predict Survival Chance"):
    st.write("probability of passenge survival chance",y)
    st.write(chance(y))