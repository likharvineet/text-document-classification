import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import gensim
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

app = Flask(__name__) # starting point of my application

# Load the model
logregmodel = pickle.load(open('model.pkl', 'rb'))
doc2vec_model = pickle.load(open('doc2vec_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    # print(np.array(list(data.values())).reshape(1, -1))
    # print(list(data.values()))
    new_data = doc2vec_model.infer_vector(word_tokenize(data))
    new_data = np.array(new_data).reshape(1, -1)
    output = logregmodel.predict(new_data)
    print(output)
    return jsonify({'prediction': str(output)})

@app.route('/predict', methods = ['POST'])
def predict():
    # data = [request.form.values()]
    label = ['Politics', 'Sports', 'Technology', 'Entertainment', 'Business']
    data = request.form['data']
    print(data)
    new_data = doc2vec_model.infer_vector(word_tokenize(data))
    new_data = np.array(new_data).reshape(1, -1)
    output = logregmodel.predict(new_data)[0]
    output = label[output]
    return render_template('home.html', prediction_text = '{}'.format(output))

if __name__ == '__main__':
    app.run(port = 5000, debug = True)