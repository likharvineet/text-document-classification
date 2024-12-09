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
import seaborn as sns
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
    # Initialize the pretrained BERT model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Example paragraphs
    paragraphs = [data]

    # Compute embeddings for the paragraphs
    embeddings = model.encode(paragraphs)

    pca_data = pca_model.transform(embeddings)
    print("pca data \n", pca_data)
    kmean_cluster = kmmeans_model.predict(pca_data)
    print("kmean cluster \n", kmean_cluster)
    doc2vec_data = np.array(doc2vec_data).reshape(1, -1)
    output = logregmodel.predict(doc2vec_data)[0]
    output = label[output]

    # Get the matplotlib plot  
    plot = get_plot(kmmeans_model, pca_data)

    
  
    # Save the figure in the static directory  
    timestamp = str(datetime.now().timestamp())
    plot.savefig(os.path.join('static', 'images', timestamp +'.png'))
    return render_template('home.html', prediction_text = '{}'.format(output), img_path = url_for('static', filename = 'images/' + timestamp + '.png'))

# ################## For testing purpose ##################
def get_plot(kmeans, new_pca_point): 
    df = pd.read_csv('pca.csv')
    
    # Define cluster names
    cluster_names = {
        0: "Politics",
        1: "Business",
        2: "Entertainment",
        3: "Sports",
        4: "Technology",
    }

    # Map cluster IDs to names
    df["cluster_name"] = df["kmeans_cluster"].map(cluster_names)

    # Calculate centroids for each cluster
    centroids = df.groupby("kmeans_cluster")[["pca1", "pca2"]].mean()

    # Plot with centroids labeled
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 8))

    # Scatter plot
    scatter = sns.scatterplot(
        x="pca1",
        y="pca2",
        hue="cluster_name",
        data=df,
        palette="viridis",
        s=100
    )

    # Annotate centroids with cluster names
    for cluster_id, (x, y) in centroids.iterrows():
        plt.text(
            x, y,
            cluster_names[cluster_id],
            fontsize=10,
            ha='center',  # Center-align text
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')  # Add a box around the text
        )
    plt.scatter(
    new_pca_point[0, 0],
    new_pca_point[0, 1],
    c="red",
    s=200,
    label="New Paragraph",
    edgecolor="black"
)

    plt.title("K-means Clustering with Cluster Names Annotated at Centroids")
    plt.xlabel("PCA Dimension 1")
    plt.ylabel("PCA Dimension 2")

    plt.tight_layout()
    return plt
   

if __name__ == '__main__':
    app.run(port = 5000, debug = True)