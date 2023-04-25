### Heat model API
# https://www.datacamp.com/tutorial/machine-learning-models-api-python


from flask import Flask, jsonify, request, redirect, url_for, flash, jsonify
# from flask_cors import CORS, cross_origin
from jinja2 import escape #pip install Jinja2==3.0.3
import json
import pandas as pd
import re
import numpy as np
from sklearn.metrics import mean_squared_error
import pickle
import datetime as dt
import matplotlib as plt
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from datetime import datetime
from flask_cors import CORS
import os
os.chdir("C:\\Users\Alvin Lim\Documents\GitHub\BC3407-Businesss-Transformation\machine learning\hazard_predictor_heat")


with open('arima502010.pkl', 'rb') as f:
    model = pickle.load(f)


app = Flask(__name__)
CORS(app)

@app.route("/HeatModel", methods=["GET", "POST"])
def HeatModel():

    # Get the input data from the request
    data = request.get_json()
    input_data = data['date']


    # date_list=input_data.split("-")
    # day=int(date_list[2].split("T")[0])
    # month=int(date_list[1])
    # year=int(date_list[0])
    # new_date=datetime(year,month,day)

    new_date = input_data.split("T")[0]
    

    

    # create machine date time variable to subtract date

    def forecast_period(d1, d2):
        d1 = datetime.strptime(d1, "%Y-%m-%d")
        d2 = datetime.strptime(d2, "%Y-%m-%d")
        return abs((d2 - d1).days)

    d1 = "2018-11-07" #last date in trainset
    d2 = new_date

    fc_period = forecast_period(d1, d2)


    # Run the machine learning model on each input data point
    pred_arima = model.forecast(fc_period)

    latest_data = pred_arima.tail(1)

    pred_out = pd.cut(x=latest_data, bins=[0,90,103,124,220],
       labels = ['Low Heat Intensity','Intermediate Heat Intensity','Danger! High Heat Intensity','Extreme Danger! Very High Heat Intensity'])

    pred_label = pred_out[pred_out.transform(type) == str].values[0]

    Heat_Intensity = [f"Predicted {pred_label} at {new_date}"]

    return(Heat_Intensity)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int("8000"))







