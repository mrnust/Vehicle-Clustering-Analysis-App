import streamlit as st  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

# Function to load and preprocess the data
def preprocess_data(file):
    data = pd.read_csv(file)
    
    data = data.dropna()
    # Replace "$null$" with NaN
    data.replace("$null$", np.nan, inplace=True)

    # Convert 'type' column to numeric
    data["type"] = pd.to_numeric(data["type"], errors='coerce')

    # Columns to convert to numeric
    numeric_columns = [
        'sales', 'resale', 'price', 'engine_s', 'horsepow', 
        'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 
        'mpg', 'lnsales'
    ]

    # Convert the columns to numeric, coercing errors to NaN
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')
    data["type"] = data["type"].apply(pd.to_numeric, errors='coerce')

    # Replace missing values in the specified columns with the mean of those columns
    for col in numeric_columns:
        data[col].fillna(data[col].mean(), inplace=True)

    # Perform one-hot encoding on 'manufact' and 'model' columns
    data = pd.get_dummies(data, columns=['manufact', 'model'])

    return data, numeric_columns

# Function to train the Isolation Forest model and perform hierarchical clustering
def train_model(data, numerical_features):
    # Normalize numerical variables
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Train Isolation Forest model
    model = IsolationForest(contamination=0.05)
    model.fit(data)

    # Predict outliers
    outliers = model.predict(data)

    # Get outlier indices
    outlier_indices = data.index[outliers == -1]

    # Remove outliers from the DataFrame
    data.drop(index=outlier_indices, inplace=True)

    # Hierarchical clustering
    dend = shc.linkage(data, method='ward')

    return dend, data, scaler

# Main function to run the Streamlit app
def main():
    st.title('Vehicle Clustering Analysis App')

    # Allow user to upload a CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

    if uploaded_file is not None:
        st.subheader('Uploaded Data')
        # Preprocess the data
        data, numerical_features = preprocess_data(uploaded_file)

        # Display the first few rows of the data
        st.write(data.head())

        # Train model and perform clustering
        dend, data, scaler = train_model(data, numerical_features)

        st.subheader("Data Dimensions")
        st.write(data.shape)

        # Plot histograms for numerical features
        st.subheader('Histograms')
        num_cols = 3
        num_rows = int(np.ceil(len(numerical_features) / num_cols))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
        axes = axes.flatten()

        for ax, col in zip(axes, numerical_features):
            ax.hist(data[col], bins=20)
            ax.set_title(col)
        
        # Remove any empty subplots
        for i in range(len(numerical_features), len(axes)):
            fig.delaxes(axes[i])

        st.pyplot(fig)

        # Plot box plots for numerical features
        st.subheader('Boxplots')
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=data[numerical_features], ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

        # Plot dendrogram
        st.subheader('Dendrogram')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('Hierarchical Clustering Dendrogram')
        dendrogram = shc.dendrogram(dend, ax=ax)
        st.pyplot(fig)

        # Number of clusters
        unique_colors = set(dendrogram['color_list'])
        number_of_clusters = len(unique_colors) - 1
        st.subheader("Number of Clusters")
        st.write(number_of_clusters)

        # Hierarchical clustering with AgglomerativeClustering
        agg_clustering = AgglomerativeClustering(n_clusters=number_of_clusters)
        agg_clustering.fit(data)

        # Retrieve the cluster labels
        cluster_labels = agg_clustering.labels_

        # Add the cluster labels to the DataFrame
        data['cluster'] = cluster_labels

        # Print the counts of each cluster
        st.subheader("Cluster Counts")
        st.write(data['cluster'].value_counts())

        # PCA for visualization
        st.subheader('PCA for Cluster Visualization')
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(data[numerical_features])
        pca_df = pd.DataFrame(data=principal_components, columns=['Component 1', 'Component 2'])
        pca_df['cluster'] = cluster_labels

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Component 1', y='Component 2', hue='cluster', data=pca_df, palette='viridis', ax=ax)
        ax.set_title('Clusters visualized using PCA')
        st.pyplot(fig)



if __name__ == "__main__":
    main()
