### Text Document Classification

This project focuses on classifying text documents using both supervised (Logistic Regression) and unsupervised (K-Means) learning techniques. It integrates machine learning with a Flask-based web application for deployment. The project also uses GitHub Actions for CI/CD and deploys the application using Docker and Heroku.

## Key Features

1. **Supervised Learning**:
   - Logistic Regression for text classification.
   - Model training and evaluation, achieving an accuracy of 93%.

2. **Unsupervised Learning**:
   - K-Means clustering for document grouping.
   - Insights into natural groupings within the data.

3. **Text Preprocessing**:
   - Cleaning, tokenization, and vectorization of text.
   - Filtering rows based on word count to reduce variability.

4. **Deployment**:
   - Flask backend for serving predictions and clustering results.
   - Docker for containerization.
   - Heroku for seamless deployment with GitHub Actions for CI/CD.

### Software and Tools Requirements

1. [Github Account](https://github.com)
2. [Heroku Account](https://heroku.com)
3. [VS Code IDE](https://code.visualstudio.com/)
4. [Git CLI](https://git-scm.com/book/en/v2/Getting-Started_The-Command-Line)

## Project Workflow

1. **Dataset**:
   - Sourced from [Kaggle](https://www.kaggle.com/).
   - Preprocessed by removing duplicates, unnecessary text, and filtering rows based on word count.

2. **Supervised Learning**:
   - Embeddings generated using **Doc2Vec**.
   - Trained a Logistic Regression model using an 80-20 train-test split.

3. **Unsupervised Learning**:
   - Embeddings created using **BERT**.
   - Dimensionality reduction applied with PCA.
   - Optimal clusters determined using the elbow method before training the K-Means model.

4. **Visualization**:
   - Word clouds to visualize the most frequent words in the dataset and each cluster.

5. **Deployment**:
   - Built a Flask API to handle predictions.
   - Automated the deployment process with GitHub Actions, Docker, and Heroku.


## Setup and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/document-classification.git
   cd document-classification

2. Create a new conda environment

    ```
    conda create -n ml python=3.11 -y
    ```
3. Activate the new conda environment

    ```
    conda activate ml
    ```

4. To install all the packages at once

    ```
    pip install -r requirements.txt
    ```
5. Run the Flask applicaiton locally:
    ```bash
    python app.py
    ```
## Result

- The Logistic Regression model achieved an accuracy of 93%.
- K-Means clustering produced meaningful groupings, visualized using word clouds and scatter plots.

## Future Improvement
    - Enhance the clustering accuracy of the unsupervised model.
    - Improve the web application's interface for better user experience.
    - Scale the deployment for handling larger datasets and concurrent users.

## Link and References
    - https://www.geeksforgeeks.org/create-scatter-charts-in-matplotlib-using-flask/
    - https://www.geeksforgeeks.org/how-to-add-graphs-to-flask-apps/
    - https://medium.com/@danielafrimi/text-clustering-using-nlp-techniques-c2e6b08b6e95
    - https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5
    - https://www.geeksforgeeks.org/doc2vec-in-nlp/
    - https://www.kaggle.com/code/alhsan/word-statistics-effect-accuracy-0-973
    - https://www.youtube.com/watch?v=MJ1vWb1rGwM&list=PLZoTAELRMXVMdvxeSuliQZcRLu3WCYVim

## Video Link
    - https://drive.google.com/file/d/1A_5T66xGOlWuW3Rf1-7VcifgA3ESNY_D/view?usp=drive_link
