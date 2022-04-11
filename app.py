from flask import Flask, request, url_for, redirect, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import pickle
import json


app =Flask(__name__)

loaded_fake_news_model = load_model('./Model/saved_models/fake_new_predictor.h5')
with open('./Model/saved_models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    body =request.json
    text = [body['news']]
    text_token = tokenizer.texts_to_sequences([text])
    text_token = pad_sequences(text_token, 700)
    out = loaded_fake_news_model.predict(x=text_token)
    print(out)
    data = {"news": None}
    if out[0][0] >= 0.5:
        data["news"] = "True News"
    else:
        data["news"] = "Fake news"

    response = app.response_class(response=json.dumps(data),
                                  status=200,
                                  mimetype='application/json')

    return response
   


if __name__ == '__main__':
    app.run('127.0.0.1', port=5500)