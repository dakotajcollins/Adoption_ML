# Load libraries
from flask import Flask, jsonify
import joblib
from flask_bootstrap import Bootstrap
from flask import render_template, redirect, url_for, request, send_from_directory, flash
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
import json
from json import JSONEncoder
import numpy
import os


#Set up Flask
TEMPLATE_DIR = os.path.abspath('templates')
STATIC_DIR = os.path.abspath('static')
# instantiate flask 
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.secret_key="Pet_Adoption"
app.config['SESSION_COOKIE_SECURE'] = False
Bootstrap(app)
clf = joblib.load("adoption_rfc_model.pkl") # Load "model.pkl"
print ('Model loaded')
model_columns = joblib.load("adoption_rfc_model_columns.pkl") # Load "model_columns.pkl"
print ('Model columns loaded')

# Home page, renders the intro.html template

@app.route('/')
def home():
    return render_template('index.html', title='homepage')

@app.route('/machinelearning')
def ML():
    return render_template('ML.html', title='MachineLearning')

@app.route('/spervised', methods=['GET', 'POST'])
def spervised():
    if request.method == 'POST':
        primary_breed= request.form['primary_breed']
        color= request.form['color']
        age= request.form['age']
        gender= request.form['gender']
        size= request.form['size']
        coat= request.form['coat']
        mix_breed= bool(request.form['mix_breed'])
        house_trained= bool(request.form['house_trained'])
        spayed_neutered= bool(request.form['spayed_neutered'])
        special_need= bool(request.form['special_need'])
        shot_current= bool(request.form['shot_current'])
        gw_childern= bool(request.form['gw_childern'])
        gw_dog= bool(request.form['gw_dog'])
        gw_cat= bool(request.form['gw_cat'])
        tag= bool(request.form['tag'])
        photo= bool(request.form['photo'])

        
        x_feature=[[primary_breed,mix_breed,color,age,gender,size,coat,house_trained,spayed_neutered,special_need,shot_current,gw_childern,gw_dog,gw_cat,tag,photo]]

        print(x_feature)

        query = pd.get_dummies(pd.DataFrame(x_feature))
        print(query)
        query = query.reindex(columns=model_columns, fill_value=0)

        prediction = list(clf.predict(query))

        print(prediction)
        image=f"static/assets/{prediction[0]}.png"
        result=prediction[0].replace("_"," ").upper()
        flash(result)
        
    return render_template('ML.html', title='spervised',image=image)





if __name__ == '__main__':

    app.run(debug=True)