import json
import pickle
from flask import Flask, app, request, jsonify, render_template
import numpy as np

app= Flask(__name__)
regmodel= pickle.load(open('regmodel.pkl', 'rb'))
scaler= pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return(render_template('home.html'))

@app.route('/predict_api', methods= ["POST"])
def predict_api():
    data= request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data= scaler.transform(np.array(list(data.values())).reshape(1,-1))
    prediction= regmodel.predict(new_data)
    print(prediction[0])
    return(jsonify(prediction[0]))

if __name__== "__main__":
    app.run(debug=True)