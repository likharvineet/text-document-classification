import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import plotly.express as px
import numpy as np
import pandas as pd
import gensim
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer


app = Flask(__name__) # starting point of my application

# Load the model
logregmodel = pickle.load(open('model.pkl', 'rb'))
doc2vec_model = pickle.load(open('doc2vec_model.pkl', 'rb'))
pca_model = pickle.load(open('pca.pkl', 'rb'))
kmmeans_model = pickle.load(open('kmean_final.pkl', 'rb'))
sentenceTransformer = pickle.load(open('sentenceTransformer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    # print(np.array(list(data.values())).reshape(1, -1))
    print(list(data.values()))
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
    print("data \n",data)
    doc2vec_data = doc2vec_model.infer_vector(word_tokenize(data))
    sentenceTransformer_data = sentenceTransformer.encode([data])
    pca_data = pca_model.transform(sentenceTransformer_data)
    print("pca data \n", pca_data)
    kmean_cluster = kmmeans_model.predict(pca_data)
    print("kmean cluster \n", kmean_cluster)
    doc2vec_data = np.array(doc2vec_data).reshape(1, -1)
    output = logregmodel.predict(doc2vec_data)[0]
    output = label[output]
    return render_template('home.html', prediction_text = '{}'.format(output))

# ################## For testing purpose ##################
def get_plot(X, kmeans): 
    pred = kmeans.fit_predict(X)
    plt.scatter(X[:,0],X[:,1],c = pred, cmap=cm.Accent)
    plt.grid(True)
    for center in kmeans.cluster_centers_:
        center = center[:2]
        plt.scatter(center[0],center[1],marker = '^',c = 'red')
    plt.scatter(6.5, 4, marker='*')
    plt.xlabel("petal length (cm)")
    plt.ylabel("petal width (cm)")
    return plt

# @app.route('/predict', methods = ['POST'])
# def predict():
#     X = np.array([[5.1, 3.5, 1.4, 0.2],
#        [4.9, 3. , 1.4, 0.2],
#        [4.7, 3.2, 1.3, 0.2],
#        [4.6, 3.1, 1.5, 0.2],
#        [5. , 3.6, 1.4, 0.2],
#        [5.4, 3.9, 1.7, 0.4],
#        [4.6, 3.4, 1.4, 0.3],
#        [5. , 3.4, 1.5, 0.2],
#        [4.4, 2.9, 1.4, 0.2],
#        [4.9, 3.1, 1.5, 0.1],
#        [5.4, 3.7, 1.5, 0.2],
#        [4.8, 3.4, 1.6, 0.2],
#        [4.8, 3. , 1.4, 0.1],
#        [4.3, 3. , 1.1, 0.1],
#        [5.8, 4. , 1.2, 0.2],
#        [5.7, 4.4, 1.5, 0.4],
#        [5.4, 3.9, 1.3, 0.4],
#        [5.1, 3.5, 1.4, 0.3],
#        [5.7, 3.8, 1.7, 0.3],
#        [5.1, 3.8, 1.5, 0.3],
#        [5.4, 3.4, 1.7, 0.2],
#        [5.1, 3.7, 1.5, 0.4],
#        [4.6, 3.6, 1. , 0.2],
#        [5.1, 3.3, 1.7, 0.5],
#        [4.8, 3.4, 1.9, 0.2],
#        [5. , 3. , 1.6, 0.2],
#        [5. , 3.4, 1.6, 0.4],
#        [5.2, 3.5, 1.5, 0.2],
#        [5.2, 3.4, 1.4, 0.2],
#        [4.7, 3.2, 1.6, 0.2],
#        [4.8, 3.1, 1.6, 0.2],
#        [5.4, 3.4, 1.5, 0.4],
#        [5.2, 4.1, 1.5, 0.1],
#        [5.5, 4.2, 1.4, 0.2],
#        [4.9, 3.1, 1.5, 0.2],
#        [5. , 3.2, 1.2, 0.2],
#        [5.5, 3.5, 1.3, 0.2],
#        [4.9, 3.6, 1.4, 0.1],
#        [4.4, 3. , 1.3, 0.2],
#        [5.1, 3.4, 1.5, 0.2],
#        [5. , 3.5, 1.3, 0.3],
#        [4.5, 2.3, 1.3, 0.3],
#        [4.4, 3.2, 1.3, 0.2],
#        [5. , 3.5, 1.6, 0.6],
#        [5.1, 3.8, 1.9, 0.4],
#        [4.8, 3. , 1.4, 0.3],
#        [5.1, 3.8, 1.6, 0.2],
#        [4.6, 3.2, 1.4, 0.2],
#        [5.3, 3.7, 1.5, 0.2],
#        [5. , 3.3, 1.4, 0.2],
#        [7. , 3.2, 4.7, 1.4],
#        [6.4, 3.2, 4.5, 1.5],
#        [6.9, 3.1, 4.9, 1.5],
#        [5.5, 2.3, 4. , 1.3],
#        [6.5, 2.8, 4.6, 1.5],
#        [5.7, 2.8, 4.5, 1.3],
#        [6.3, 3.3, 4.7, 1.6],
#        [4.9, 2.4, 3.3, 1. ],
#        [6.6, 2.9, 4.6, 1.3],
#        [5.2, 2.7, 3.9, 1.4],
#        [5. , 2. , 3.5, 1. ],
#        [5.9, 3. , 4.2, 1.5],
#        [6. , 2.2, 4. , 1. ],
#        [6.1, 2.9, 4.7, 1.4],
#        [5.6, 2.9, 3.6, 1.3],
#        [6.7, 3.1, 4.4, 1.4],
#        [5.6, 3. , 4.5, 1.5],
#        [5.8, 2.7, 4.1, 1. ],
#        [6.2, 2.2, 4.5, 1.5],
#        [5.6, 2.5, 3.9, 1.1],
#        [5.9, 3.2, 4.8, 1.8],
#        [6.1, 2.8, 4. , 1.3],
#        [6.3, 2.5, 4.9, 1.5],
#        [6.1, 2.8, 4.7, 1.2],
#        [6.4, 2.9, 4.3, 1.3],
#        [6.6, 3. , 4.4, 1.4],
#        [6.8, 2.8, 4.8, 1.4],
#        [6.7, 3. , 5. , 1.7],
#        [6. , 2.9, 4.5, 1.5],
#        [5.7, 2.6, 3.5, 1. ],
#        [5.5, 2.4, 3.8, 1.1],
#        [5.5, 2.4, 3.7, 1. ],
#        [5.8, 2.7, 3.9, 1.2],
#        [6. , 2.7, 5.1, 1.6],
#        [5.4, 3. , 4.5, 1.5],
#        [6. , 3.4, 4.5, 1.6],
#        [6.7, 3.1, 4.7, 1.5],
#        [6.3, 2.3, 4.4, 1.3],
#        [5.6, 3. , 4.1, 1.3],
#        [5.5, 2.5, 4. , 1.3],
#        [5.5, 2.6, 4.4, 1.2],
#        [6.1, 3. , 4.6, 1.4],
#        [5.8, 2.6, 4. , 1.2],
#        [5. , 2.3, 3.3, 1. ],
#        [5.6, 2.7, 4.2, 1.3],
#        [5.7, 3. , 4.2, 1.2],
#        [5.7, 2.9, 4.2, 1.3],
#        [6.2, 2.9, 4.3, 1.3],
#        [5.1, 2.5, 3. , 1.1],
#        [5.7, 2.8, 4.1, 1.3],
#        [6.3, 3.3, 6. , 2.5],
#        [5.8, 2.7, 5.1, 1.9],
#        [7.1, 3. , 5.9, 2.1],
#        [6.3, 2.9, 5.6, 1.8],
#        [6.5, 3. , 5.8, 2.2],
#        [7.6, 3. , 6.6, 2.1],
#        [4.9, 2.5, 4.5, 1.7],
#        [7.3, 2.9, 6.3, 1.8],
#        [6.7, 2.5, 5.8, 1.8],
#        [7.2, 3.6, 6.1, 2.5],
#        [6.5, 3.2, 5.1, 2. ],
#        [6.4, 2.7, 5.3, 1.9],
#        [6.8, 3. , 5.5, 2.1],
#        [5.7, 2.5, 5. , 2. ],
#        [5.8, 2.8, 5.1, 2.4],
#        [6.4, 3.2, 5.3, 2.3],
#        [6.5, 3. , 5.5, 1.8],
#        [7.7, 3.8, 6.7, 2.2],
#        [7.7, 2.6, 6.9, 2.3],
#        [6. , 2.2, 5. , 1.5],
#        [6.9, 3.2, 5.7, 2.3],
#        [5.6, 2.8, 4.9, 2. ],
#        [7.7, 2.8, 6.7, 2. ],
#        [6.3, 2.7, 4.9, 1.8],
#        [6.7, 3.3, 5.7, 2.1],
#        [7.2, 3.2, 6. , 1.8],
#        [6.2, 2.8, 4.8, 1.8],
#        [6.1, 3. , 4.9, 1.8],
#        [6.4, 2.8, 5.6, 2.1],
#        [7.2, 3. , 5.8, 1.6],
#        [7.4, 2.8, 6.1, 1.9],
#        [7.9, 3.8, 6.4, 2. ],
#        [6.4, 2.8, 5.6, 2.2],
#        [6.3, 2.8, 5.1, 1.5],
#        [6.1, 2.6, 5.6, 1.4],
#        [7.7, 3. , 6.1, 2.3],
#        [6.3, 3.4, 5.6, 2.4],
#        [6.4, 3.1, 5.5, 1.8],
#        [6. , 3. , 4.8, 1.8],
#        [6.9, 3.1, 5.4, 2.1],
#        [6.7, 3.1, 5.6, 2.4],
#        [6.9, 3.1, 5.1, 2.3],
#        [5.8, 2.7, 5.1, 1.9],
#        [6.8, 3.2, 5.9, 2.3],
#        [6.7, 3.3, 5.7, 2.5],
#        [6.7, 3. , 5.2, 2.3],
#        [6.3, 2.5, 5. , 1.9],
#        [6.5, 3. , 5.2, 2. ],
#        [6.2, 3.4, 5.4, 2.3],
#        [5.9, 3. , 5.1, 1.8]])
#     data = [float(x) for x in request.form.values()]
#     print(data)
#     data = np.array(data).reshape(1, -1)
#     cluster = kmmeans_model.cluster_centers_
    
#     print(cluster)
#     output = kmmeans_model.predict(data)[0]
    

#     # Get the matplotlib plot  
#     plot = get_plot(X, kmmeans_model)

    
  
#     # Save the figure in the static directory  
#     timestamp = str(datetime.now().timestamp())
#     plot.savefig(os.path.join('static', 'images', timestamp +'.png'))



#     return render_template('home.html', prediction_text = '{}'.format(output), img_path = url_for('static', filename = 'images/' + timestamp + '.png'))
   

if __name__ == '__main__':
    app.run(port = 5000, debug = True)