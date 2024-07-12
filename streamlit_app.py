import streamlit as st
import pandas as pd
import numpy as np


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
