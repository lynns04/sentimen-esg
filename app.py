import streamlit as st
import joblib
import pandas as pd

# Memuat model yang telah dilatih
model = joblib.load('sentiment_model.pkl')

# Fungsi untuk prediksi sentimen
def predict_sentiment(text):
    # Langsung menggunakan model tanpa preprocessing tambahan
    result = model.predict([text])
    return result[0]  # Sesuaikan dengan format output model

# UI Streamlit
st.title("Dashboard Analisis Sentimen ESG")

# Input teks dari pengguna
user_input = st.text_area("Masukkan teks untuk analisis sentimen")

if user_input:
    # Melakukan analisis sentimen
    sentiment = predict_sentiment(user_input)
    # Menampilkan hasil analisis sentimen
    st.write(f"Sentimen: {sentiment}")

# File upload
uploaded_file = st.file_uploader("Upload file untuk analisis sentimen", type=["txt", "csv"])

if uploaded_file:
    if uploaded_file.type == "text/plain":
        # Membaca file teks
        text = uploaded_file.read().decode("utf-8")
        # Melakukan analisis sentimen pada teks
        sentiment = predict_sentiment(text)
        # Menampilkan hasil analisis sentimen
        st.write(f"Sentimen: {sentiment}")
    elif uploaded_file.type == "text/csv":
        # Membaca file CSV
        df = pd.read_csv(uploaded_file)
        # Memastikan kolom pertama adalah teks untuk analisis
        if df.shape[1] > 0:
            df['sentiment'] = df.iloc[:, 0].apply(lambda x: predict_sentiment(str(x)))
            st.write(df)
        else:
            st.error("File CSV tidak memiliki kolom yang sesuai untuk analisis sentimen.")
