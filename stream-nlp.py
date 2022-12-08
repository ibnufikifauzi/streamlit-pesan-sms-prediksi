import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

#load save model
model_fraud = pickle.load(open('model_fraud.sav', 'rb'))

tfidf = TfidfVectorizer

loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("new_selected_feature_tf-idf.sav", "rb"))))


#Judul Halaman
st.title('Prediksi Pesan SMS')

clean_teks = st.text_input('Masukan Teks Pesan')

fraud_detection = ''

if st.button('Hasil Prediksi'):
    predict_fraud = model_fraud.predict(loaded_vec.fit_transform([clean_teks]))

    if (predict_fraud == 0):
        fraud_detection = 'Pesan Normal'
    elif (predict_fraud == 1):
        fraud_detection = 'Pesan Penipuan'
    else:
        fraud_detection = 'Pesan Promo'

st.success(fraud_detection)
