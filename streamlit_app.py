import streamlit as st
import numpy as np
from minisom import MiniSom
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import io

st.title("Klaster K-means & SOM")
st.write("Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/).")

# Fungsi untuk memuat file python yang diunggah
def load_script(file_path):
    with open(file_path, 'r') as file:
        script = file.read()
    return script

# Fungsi untuk menjalankan script Python yang diunggah
def run_script(script, globals_dict):
    exec(script, globals_dict)

# Judul aplikasi
st.title("Aplikasi Eksekusi Program dari File Python")

# Unggah file Python
uploaded_file = st.file_uploader("Unggah file Python (.py)", type="py")
uploaded_excel = st.file_uploader("Unggah file Excel/CSV", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Simpan file Python yang diunggah
    script = load_script(uploaded_file)

    if uploaded_excel is not None:
        # Baca file Excel/CSV
        if uploaded_excel.name.endswith('.csv'):
            df = pd.read_csv(uploaded_excel)
        else:
            df = pd.read_excel(uploaded_excel)

        st.write("File yang diunggah:")
        st.write(df)

        # Jalankan script Python yang diunggah
        globals_dict = {'uploaded_df': df}  # Ini adalah variabel global yang bisa digunakan dalam script
        run_script(script, globals_dict)

        # Tampilkan hasil yang dihasilkan oleh script
        if 'output' in globals_dict:
            st.write("Hasil:")
            st.write(globals_dict['output'])
        else:
            st.write("Script tidak menghasilkan variabel 'output'")
    else:
        st.warning("Unggah file Excel atau CSV untuk dijalankan dengan script")

# Check if data is available
if uploaded_excel is not None:
    df = pd.read_csv(uploaded_excel) if uploaded_excel.name.endswith('.csv') else pd.read_excel(uploaded_excel)
    
    # Selecting relevant features
    features = df[['Age', 'Interest_Soccer', 'Interest_Swimming', 'Interest_Volleyball']]

    # Standardizing the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Normalizing the data
    normalizer = MinMaxScaler()
    normalized_features = normalizer.fit_transform(scaled_features)

    # Using PCA to reduce data dimensions
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(normalized_features)

    # Finding the best number of clusters for K-Means
    best_kmeans_score = -1
    best_kmeans_params = None
    best_kmeans_labels = None

    for n_clusters in range(2, 10):  # Limiting the range for speed
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

    # Finding the best number of clusters for SOM
    best_som_score = -1
    best_som_params = None
    best_som_labels = None

    for x in range(2, 10):  # Limiting the range for speed
        for y in range(2, 10):
            som = MiniSom(x=x, y=y, input_len=2, sigma=1.0, learning_rate=0.5)
            som.random_weights_init(pca_features)
            som.train_random(pca_features, 100)

            # Fixing SOM labeling
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

    # Plotting the scatter diagram for K-Means and SOM based on sports interest
    sports = ['Interest_Soccer', 'Interest_Swimming', 'Interest_Volleyball']
    colors = ['green', 'blue', 'orange']  # Colors for each sport
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))

    for i, (sport, color) in enumerate(zip(sports, colors)):
        # Plot for K-Means
        axes[i, 0].scatter(pca_features[:, 0], pca_features[:, 1], c=best_kmeans_labels, cmap='viridis', alpha=0.5, edgecolor='black')
        axes[i, 0].set_title(f'K-Means Clustering: PCA Component 1 vs PCA Component 2')
        axes[i, 0].set_xlabel('PCA Component 1')
        axes[i, 0].set_ylabel('PCA Component 2')

        # Plot for SOM
        axes[i, 1].scatter(pca_features[:, 0], pca_features[:, 1], c=best_som_labels, cmap='viridis', alpha=0.5, edgecolor='black')
        axes[i, 1].set_title(f'SOM Clustering: PCA Component 1 vs PCA Component 2')
        axes[i, 1].set_xlabel('PCA Component 1')
        axes[i, 1].set_ylabel('PCA Component 2')

    plt.tight_layout()
    st.pyplot(fig)
