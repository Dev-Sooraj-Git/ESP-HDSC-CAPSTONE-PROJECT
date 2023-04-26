import numpy as np
import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

from PIL import Image
pickle_in = open("ESPmodel.pkl", "rb")
model = pickle.load(pickle_in)

df = pd.read_csv("Electrical_H_DB")
country_dict = {country: i for i, country in enumerate(df['countryname'].unique())}
scaler = StandardScaler()
df['scaled_elecrate_rural'] = scaler.fit_transform(df[['elecrate_rural']]) 

# creating a dictionary for electrate_rural with scalled elecrate rural
Scale_dict = dict(zip(df['elecrate_rural'], df['scaled_elecrate_rural']))

# @app.route('/')
def welcome():
    return "Welcome All"

def predict_in_cost(country_label, Sc_converter):

    prediction = model.predict([[country_label, Sc_converter]])
    print(prediction)
    return prediction

rad =st.sidebar.radio("Navigation",["Home","About Us"])
def main():
    st.title("Electricity Supply Prediction")
    #html_temp = """
    # <div style="background-color:tomato;padding:5px">
    # <h2 style="color:white;text-align:center;"> GLOBAL HOUSEHOLD ELECTRIFICATION PREDICTION ML APP </h2>
    # </div>
    # <marquee> Welcome To Electrical Supply Prediction, Have a Good Day </marquee>
    html_temp = """    
        <title>Global Household Electrification Prediction ML App</title>
        <style>
            title {
                font-family: Arial, sans-serif;
            
                padding: 5px;
            }
                    
            .header {
                background-color: #0F4C81;
                padding: 5px;
            }
            .header h2 {
                color: white;
                text-align: center;
                margin: 0;
            }

        </style>
        </head>
        <body>
        <div class="header">
            <h2>Global Household Electrification Prediction ML App</h2>
        </div>
        </body>
    
    """

    if rad == "Home":
        st.markdown(html_temp, unsafe_allow_html=True)
    
        countryname = st.selectbox("Select Country Name:", df['countryname'].unique())
        elecrate_rural = st.selectbox("Electrification Rate:",df['elecrate_rural'].unique() )
        result = ""

        if st.button("Predict"):
            country_label = country_dict[countryname]
            Sc_converter = Scale_dict[elecrate_rural]
            result =predict_in_cost(country_label, Sc_converter)
        st.success('Electricity Supply In Selected Area Is : {}'.format(result))

    if rad == "About Us":
        if st.button("About"):
            st.text("Author : Sooraj S ")
            st.text("model used: RandomForestClassifier")
            st.text("Built with Streamlit")
            st.text("This project is performed under hamoye data science internship")
            st.text("HDSC_23  CAPSTONE Project ; Team Django")
            #st.text("Licenced To : Team Django")

if __name__ == '__main__':
    main()
