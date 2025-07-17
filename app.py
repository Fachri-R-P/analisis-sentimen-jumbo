import streamlit as st
import pickle
import re
import joblib

st.markdown(
    """
    <style>
    .custom-bg::before {
        content: "";
        background: rgba(0, 0, 0, 0.5); /* Lapisan hitam semi transparan */
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: 0;
    }

    .custom-bg {
        background-image: url("https://springboard.id/wp-content/uploads/2025/02/jumbo-3.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: -2;
    }

    .stApp {
        background-color: transparent;
    }
    </style>

    <div class="custom-bg"></div>
    """,
    unsafe_allow_html=True
)


# Load model dan vectorizer
model = joblib.load('model_nb.pkl')
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')


# Fungsi preprocessing (cleaning + normalisasi bahasa tidak baku)
def clean_text(text):
    text = re.sub(r"@[A-Za-z0-9_]+", '', text)
    text = re.sub(r"#\w+", '', text)
    text = re.sub(r"RT[\s]+", '', text)
    text = re.sub(r"https?://\S+", '', text)
    text = re.sub(r"[^A-Za-z0-9\s]", '', text)
    text = re.sub(r'(.)\1+', r'\1', text)


    # Normalisasi kata tidak baku
    text = re.sub(r'\bpls\b', 'tolong', text)
    text = re.sub(r'\bthx\b', 'terima kasih', text)
    text = re.sub(r'\bbur\b', 'kamu', text)
    text = re.sub(r'\bbu\b', 'kamu', text)
    text = re.sub(r'\bbim\b', 'saya', text)
    text = re.sub(r'\bga\b', 'tidak', text)
    text = re.sub(r'\byg\b', 'yang', text)
    text = re.sub(r'\bdg\b', 'dengan', text)
    text = re.sub(r'\btdk\b', 'tidak', text)
    text = re.sub(r'\baja\b', 'saja', text)
    text = re.sub(r'\bgak\b', 'tidak', text)
    text = re.sub(r'\bdlm\b', 'dalam', text)
    text = re.sub(r'\bkl\b', 'kalau', text)
    text = re.sub(r'\bkbrn\b', 'karena', text)
    text = re.sub(r'\bdri\b', 'dari', text)
    text = re.sub(r'\bngg\b', 'ingin', text)
    text = re.sub(r'\bdk\b', 'tidak', text)
    text = re.sub(r'\bsbg\b', 'sebagai', text)
    text = re.sub(r'\bbmlm\b', 'belum', text)
    text = re.sub(r'\bsdh\b', 'sudah', text)
    text = re.sub(r'\baj\b', 'saja', text)
    text = re.sub(r'\bknp\b', 'kenapa', text)
    text = re.sub(r'\bnih\b', 'ini', text)
    text = re.sub(r'\bdeh\b', 'sudah', text)
    text = re.sub(r'\bbok\b', 'baik', text)
    text = re.sub(r'\bken\b', 'kenapa', text)
    text = re.sub(r'\blg\b', 'lagi', text)
    text = re.sub(r'\bsmg\b', 'semoga', text)
    text = text.lower()
    return text

# Judul Aplikasi
st.title("🔍 Analisis Sentimen Film Jumbo")
st.write("Masukkan komentar dan sistem akan memprediksi apakah sentimennya **positif** atau **negatif**.")

# Input pengguna
user_input = st.text_area("📝 Masukkan komentar:")

# Tombol Analisis
if st.button("🔍 Analisis"):
    if user_input.strip() == "":
        st.warning("Silakan masukkan komentar terlebih dahulu.")
    else:
        # Preprocessing
        cleaned = clean_text(user_input)

        # Transformasi dengan TF-IDF
        vectorized = vectorizer.transform([cleaned])

        # Prediksi dengan model
        prediction = model.predict(vectorized)

        decoded_prediction = label_encoder.inverse_transform(prediction)

        # Tampilkan hasil
        st.info(f"Setelah preprocessing:\n\n{cleaned}")
        if decoded_prediction[0] == 'Positif':
            st.success("✅ Sentimen: POSITIF")
        else:
            st.error("❌ Sentimen: NEGATIF")

