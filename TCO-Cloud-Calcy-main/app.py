from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('cloud_costs.csv')

# Remove leading/trailing whitespaces in column names
df.columns = df.columns.str.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    # Add any necessary logic here if needed
    return render_template('result.html', top_providers=["Provider1", "Provider2", "Provider3"])

@app.route('/awareness')
def awareness():
    # Add any necessary logic here if needed
    return render_template('awareness.html')

@app.route('/index2')
def index2():
    return render_template('index2.html')

@app.route('/login')
def login():
    return render_template('loginpage2.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user inputs
    iops = float(request.form['iops'])
    vcpus = float(request.form['vcpus'])
    ram = float(request.form['ram'])
    bandwidth = float(request.form['bandwidth'])
    backup_ratio = float(request.form['backup_ratio'])
    dedupe_ratio = float(request.form['dedupe_ratio'])

    # Preprocess user inputs
    user_inputs = np.array([[iops, vcpus, ram, bandwidth, backup_ratio, dedupe_ratio]])
    scaler = MinMaxScaler()
    user_inputs_scaled = scaler.fit_transform(user_inputs)

    # Placeholder for machine learning model
    kmeans = KMeans(n_clusters=3, random_state=42)

    # Use the same column names as in the DataFrame
    columns_to_use = ['IOPS', 'vCPUs', 'RAM (GB)', 'Network Bandwidth (Gbps)', 'Backup Ratio', 'Dedupe Ratio']

    df['Cluster'] = kmeans.fit_predict(df[columns_to_use])
    user_cluster = kmeans.predict(user_inputs_scaled)[0]

    # Get top 3 cloud providers for the user's cluster
    top_providers = df[df['Cluster'] == user_cluster].sort_values('Total Cost for 2 Years (USD)').head(3)['Cloud Provider'].tolist()

    # Pass the df DataFrame to the template context
    return render_template('result.html', top_providers=top_providers, df=df)

if __name__ == '__main__':
    app.run(debug=True)
