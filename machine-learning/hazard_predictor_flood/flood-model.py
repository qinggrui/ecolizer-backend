

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import pickle

app = Flask(__name__)
CORS(app) 



model = pickle.load(open("flood.pkl", "rb"))

#Create the web address which hosts the API
#method POST means use "body" on Postman, if GET then use params
@app.route("/predict", methods = ['POST']) #This adds a "/predict" to the end of the website
def predict(): #This is to query for the input from the user
    #Request for input

    data = request.get_json()

    input_year = int(data['year'])
    input_month = int(data['month'])
    input_maxtemp = int(data['maxtemp'])
    input_mintemp = int(data['mintemp'])
    input_rainfall = int(data['rainfall'])
    input_humidity = int(data['humidity'])
    input_windspeed = int(data['speed'])
    input_cloudcoverage = int(data['cloud'])
    input_sunshine = int(data['sunshinebrightness'])
    input_latitude = int(data['latitude'])
    input_longitude = int(data['longitude'])
    input_altitude = int(data['altitude'])

# {
#     "year": 2023,
#     "month": 2,
#     "max_temp": 34,
#     "min_temp": 20,
#     "rainfall": 120,
#     "humidity": 60,
#     "wind_speed": 1,
#     "cloud_coverage": 2,
#     "sunshine": 8,
#     "latitude": 20,
#     "longitude": 90,
#     "altitude": 20
# }
    test = model.predict([[input_year, input_month, input_maxtemp, input_mintemp, input_rainfall, input_humidity,
                           input_windspeed, input_cloudcoverage, input_sunshine, input_latitude, input_longitude, input_altitude]]) #Predicting based on the input

    test_result = "No Flood Risk Predicted" if test == 0 else "Predicted Risk of Flood" 
    return jsonify(test_result) #Returns the result in json format (jsonify makes it into json)

#Inputs should come in json format
if __name__ == '__main__':
    app.run() #Default port is 5000, can add port = ? to edit the port to your desired port



#Steps:
# Importing libraries --> Done
# Load the machine learning model --> Done
# Build functions to preprocess and to predict the image --> Feature extraction or what?
# Initialize the flask object --> I think it's done?
# Set the route and the function that returns something to the user’s browser --> I think done?
# Run and test the API --> How?