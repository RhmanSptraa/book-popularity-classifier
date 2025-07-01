import streamlit as st
import joblib
import numpy as np

# Load model dan encoder
model = joblib.load("model.pkl")
le_authors = joblib.load("le_authors.pkl")
le_lang = joblib.load("le_lang.pkl")
le_target = joblib.load("le_target.pkl")

st.set_page_config(page_title="Prediksi Popularitas Buku", layout="centered")

st.title("📚 Prediksi Popularitas Buku")
st.write("Masukkan informasi buku untuk memprediksi apakah buku tersebut populer atau tidak.")

# Input form
average_rating = st.slider("Rata-rata Rating", 0.0, 5.0, 3.5, step=0.1)
text_reviews_count = st.number_input("Jumlah Ulasan Teks", min_value=0, value=100)
num_pages = st.number_input("Jumlah Halaman", min_value=1, value=200)
authors = st.text_input("Nama Penulis", value="J.K. Rowling")
language_code = st.text_input("Kode Bahasa (contoh: eng)", value="eng")

if st.button("Prediksi"):
    try:
        # Transform input
        author_encoded = le_authors.transform([authors])[0]
    except:
        st.warning("Penulis belum dikenal model, akan menggunakan default.")
        author_encoded = 0

    try:
        lang_encoded = le_lang.transform([language_code])[0]
    except:
        st.warning("Kode bahasa belum dikenal model, akan menggunakan default.")
        lang_encoded = 0

    # Buat array fitur
    input_data = np.array([[average_rating, num_pages, author_encoded, lang_encoded, text_reviews_count]])

    # Prediksi
    prediction = model.predict(input_data)[0]
    label = le_target.inverse_transform([prediction])[0]

    st.success(f"📖 Buku ini diprediksi sebagai: **{label}**")
