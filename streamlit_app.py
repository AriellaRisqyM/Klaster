
import streamlit as st
import numpy as np
from minisom import MiniSom
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


st.title("Klaster K-means & SOM")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

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

# Memilih fitur yang relevan
features = data[['Age', 'Interest_Soccer', 'Interest_Swimming', 'Interest_Volleyball']]

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

    print(f"K-Means with {n_clusters} clusters: Silhouette Score = {kmeans_silhouette}, Davies-Bouldin Index = {kmeans_davies_bouldin}")

    if kmeans_silhouette > best_kmeans_score:
        best_kmeans_score = kmeans_silhouette
        best_kmeans_params = (n_clusters, kmeans_silhouette, kmeans_davies_bouldin)
        best_kmeans_labels = kmeans_labels

# Print the best parameters found
print("\nBest K-Means Parameters:")
print(f"Number of Clusters: {best_kmeans_params[0]}")
print(f"Silhouette Score: {best_kmeans_params[1]}")
print(f"Davies-Bouldin Index: {best_kmeans_params[2]}")

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
        som_labels = np.ravel_multi_index(winner_coordinates, (x,y))

        # Check for number of unique labels
        n_unique_labels = len(np.unique(som_labels))
        if n_unique_labels <= 1 or n_unique_labels >= len(pca_features):
            print(f"SOM with grid {x}x{y}: Insufficient unique labels ({n_unique_labels})")
            continue  # Skip to the next iteration

        som_silhouette = silhouette_score(pca_features, som_labels)
        som_davies_bouldin = davies_bouldin_score(pca_features, som_labels)

        print(f"SOM with grid {x}x{y}: Silhouette Score = {som_silhouette}, Davies-Bouldin Index = {som_davies_bouldin}")

        if som_silhouette > best_som_score:
            best_som_score = som_silhouette
            best_som_params = (x, y, som_silhouette, som_davies_bouldin)
            best_som_labels = som_labels

print("\nBest SOM Parameters:")
if best_som_params is not None:  # Check if any valid SOM parameters were found
    print(f"Grid Size: {best_som_params[0]}x{best_som_params[1]}")
    print(f"Silhouette Score: {best_som_params[2]}")
    print(f"Davies-Bouldin Index: {best_som_params[3]}")
else:
    print("No valid SOM parameters found.")

# Diagram persebaran untuk K-Means dan SOM berdasarkan minat olahraga
sports = ['Interest_Soccer', 'Interest_Swimming', 'Interest_Volleyball']
colors = ['green', 'blue', 'orange']  # Warna untuk masing-masing olahraga
plt.figure(figsize=(18, 12))

for i, (sport, color) in enumerate(zip(sports, colors)):
    # Plot untuk K-Means
    plt.subplot(3, 2, 2*i + 1)
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=best_kmeans_labels, cmap='viridis', alpha=0.5, edgecolor='black')
    plt.title(f'K-Means Clustering: PCA Component 1 vs PCA Component 2')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    # Plot untuk SOM
    plt.subplot(3, 2, 2*i + 2)
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=best_som_labels, cmap='viridis', alpha=0.5, edgecolor='black')
    plt.title(f'SOM Clustering: PCA Component 1 vs PCA Component 2')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

plt.tight_layout()
plt.show()
