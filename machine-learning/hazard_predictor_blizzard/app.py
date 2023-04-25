### Blizzard model API
# https://www.datacamp.com/tutorial/machine-learning-models-api-python


from flask import Flask, jsonify, request, redirect, url_for, flash, jsonify
# from flask_cors import CORS, cross_origin
from jinja2 import escape #pip install Jinja2==3.0.3
import json
import pandas as pd
import re
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pickle
from flask_cors import CORS

with open('model.pkl' , 'rb') as i:
    model = pickle.load(i)


app = Flask(__name__)
CORS(app)

@app.route("/blizzmodel", methods=["GET","POST"])

# hazard types and keywords to scrape tweets for

def BlizzModel():

    # Get the input data from the request

    blizzparams = request.get_json()

    # {"Temperature" : 30,
    #     "Relative_Humidity": 30,
    #     "Wind_Gust": 30,
    #     "Wind_Speed": 30,de
    #     "Wind_Direction": 30,
    #     "Cloud_Cover": 30,
    #     "Sunshine": 30,
    #     "Sea_Level_Pressure": 30,
    #     "Vapor_Pressure_Deficit": 30
    #     }

    Temperature = int(blizzparams["temperature"])
    Relative_Humidity = int(blizzparams["humidity"])
    Wind_Gust = int(blizzparams["gust"])
    Wind_Speed = int(blizzparams["speed"])
    Wind_Direction = int(blizzparams["direction"])
    Cloud_Cover = int(blizzparams["cloud"])
    Sunshine = int(blizzparams["sunshineduration"])
    Sea_Level_Pressure = int(blizzparams["pressure"])
    Vapor_Pressure_Deficit = int(blizzparams["vapor"])


    # Run the machine learning model on each input data point
    print(np.array([[Temperature, Relative_Humidity, Wind_Gust, Wind_Speed, Wind_Direction,
                              Cloud_Cover, Sunshine, Sea_Level_Pressure, Vapor_Pressure_Deficit]]))
    

    pred_xg = model.predict(np.array([[Temperature, Relative_Humidity, Wind_Gust, Wind_Speed, Wind_Direction,
                              Cloud_Cover, Sunshine, Sea_Level_Pressure, Vapor_Pressure_Deficit]]))
    print("gayfag")
    Snow_Intensity = [f"Normal snow with Snow Depth of {pred_xg[0]:0.3f} mm" if pred_xg < 2 else
                        f"Moderate snow with Snow Depth of {pred_xg[0]:0.3f} mm" if pred_xg <5 else
                        f"Blizzard/Heavy Snow with Snow Depth of {pred_xg[0]:0.3f} mm" if pred_xg >= 5 else 'NA']

    return(Snow_Intensity[0])

if __name__ == "__main__":
    app.run(debug=True)





