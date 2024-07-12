import streamlit as st
import numpy as np
from minisom import MiniSom
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Ensure required package is installed
try:
    import openpyxl
except ImportError:
    st.error("Please install openpyxl to read Excel files: pip install openpyxl")
    st.stop()

st.title("Klaster K-means & SOM")
st.write("Unggah file CSV/Excel Anda untuk memulai klasterisasi.")

# Unggah file Excel/CSV
uploaded_file = st.file_uploader("Unggah file Excel/CSV", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Baca file Excel/CSV
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("File yang diunggah:")
    st.write(df)

    # Memilih fitur yang relevan
    features = df[['Age', 'Interest_Soccer', 'Interest_Swimming', 'Interest_Volleyball']]

    # Melakukan scaling pada data dengan StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Melakukan normalisasi pada data dengan MinMaxScaler
    normalizer = MinMaxScaler()
    normalized_features = normalizer.fit_transform(scaled_features)

    # Menggunakan PCA untuk mengurangi dimensi data
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(normalized_features)

    # Mencari jumlah klaster terbaik untuk K-Means dengan data yang sudah dinormalisasi
    best_kmeans_score = -1
    best_kmeans_params = None
    best_kmeans_labels = None

    for n_clusters in range(2, 10):  # Membatasi range untuk kecepatan
        kmeans = KMeans(n_clusters=n_clusters, random_state=50)
        kmeans_labels = kmeans.fit_predict(pca_features)
        kmeans_silhouette = silhouette_score(pca_features, kmeans_labels)
        kmeans_davies_bouldin = davies_bouldin_score(pca_features, kmeans_labels)

        if kmeans_silhouette > best_kmeans_score:
            best_kmeans_score = kmeans_silhouette
            best_kmeans_params = (n_clusters, kmeans_silhouette, kmeans_davies_bouldin)
            best_kmeans_labels = kmeans_labels

    st.write("\nBest K-Means Parameters:")
    st.write(f"Number of Clusters: {best_kmeans_params[0]}")
    st.write(f"Silhouette Score: {best_kmeans_params[1]}")
    st.write(f"Davies-Bouldin Index: {best_kmeans_params[2]}")

    # Mencari jumlah klaster terbaik untuk SOM dengan data yang sudah dinormalisasi
    best_som_score = -1
    best_som_params = None
    best_som_labels = None

    for x in range(2, 10):  # Membatasi range untuk kecepatan
        for y in range(2, 10):
            som = MiniSom(x=x, y=y, input_len=2, sigma=1.0, learning_rate=0.5)
            som.random_weights_init(pca_features)
            som.train_random(pca_features, 100)

            # Memperbaiki pemberian label SOM
            winner_coordinates = np.array([som.winner(x) for x in pca_features]).T
            som_labels = np.ravel_multi_index(winner_coordinates, (x, y))

            # Check for number of unique labels
            n_unique_labels = len(np.unique(som_labels))
            if n_unique_labels <= 1 or n_unique_labels >= len(pca_features):
                continue  # Skip to the next iteration

            som_silhouette = silhouette_score(pca_features, som_labels)
            som_davies_bouldin = davies_bouldin_score(pca_features, som_labels)

            if som_silhouette > best_som_score:
                best_som_score = som_silhouette
                best_som_params = (x, y, som_silhouette, som_davies_bouldin)
                best_som_labels = som_labels

    st.write("\nBest SOM Parameters:")
    if best_som_params is not None:  # Check if any valid SOM parameters were found
        st.write(f"Grid Size: {best_som_params[0]}x{best_som_params[1]}")
        st.write(f"Silhouette Score: {best_som_params[2]}")
        st.write(f"Davies-Bouldin Index: {best_som_params[3]}")
    else:
        st.write("No valid SOM parameters found.")

    # Diagram persebaran untuk K-Means dan SOM berdasarkan minat olahraga
    sports = ['Interest_Soccer', 'Interest_Swimming', 'Interest_Volleyball']
    colors = ['green', 'blue', 'orange']  # Warna untuk masing-masing olahraga
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))

    for i, (sport, color) in enumerate(zip(sports, colors)):
        # Plot untuk K-Means
        axes[i, 0].scatter(pca_features[:, 0], pca_features[:, 1], c=best_kmeans_labels, cmap='viridis', alpha=0.5, edgecolor='black')
        axes[i, 0].set_title(f'K-Means Clustering: PCA Component 1 vs PCA Component 2')
        axes[i, 0].set_xlabel('PCA Component 1')
        axes[i, 0].set_ylabel('PCA Component 2')

        # Plot untuk SOM
        axes[i, 1].scatter(pca_features[:, 0], pca_features[:, 1], c=best_som_labels, cmap='viridis', alpha=0.5, edgecolor='black')
        axes[i, 1].set_title(f'SOM Clustering: PCA Component 1 vs PCA Component 2')
        axes[i, 1].set_xlabel('PCA Component 1')
        axes[i, 1].set_ylabel('PCA Component 2')

    plt.tight_layout()
    st.pyplot(fig)
