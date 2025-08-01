import streamlit as st

st.title("Modèle Word2Vec")

# Chargement du modèle
from tensorflow.keras.models import load_model
model = load_model("word2vec.h5")

# Chargement du vocabulaire
import joblib
word2idx, idx2word,vocab_size = joblib.load("vocab_movieReview")

# Récupération de la matrice embedding
vectors = model.layers[0].trainable_weights[0].numpy()
st.subheader("Matrice Embedding du vocabulaire")
st.dataframe(data=vectors)

import numpy as np
from sklearn.preprocessing import Normalizer

def dot_product(vec1, vec2):
    return np.sum((vec1*vec2))

def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2)/np.sqrt(dot_product(vec1, vec1)*dot_product(vec2, vec2))

def find_closest(word_index, vectors, number_closest):
    list1=[]
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def compare(index_word1, index_word2, index_word3, vectors, number_closest):
    list1=[]
    query_vector = vectors[index_word1] - vectors[index_word2] + vectors[index_word3]
    normalizer = Normalizer()
    query_vector =  normalizer.fit_transform([query_vector], 'l2')
    query_vector= query_vector[0]
    for index, vector in enumerate(vectors):
        if not np.array_equal(vector, query_vector):
            dist = cosine_similarity(vector, query_vector)
            list1.append([dist,index])
    return np.asarray(sorted(list1,reverse=True)[:number_closest])

def print_closest(word, number=10):
    index_closest_words = find_closest(word2idx[word], vectors, number)
    for index_word in index_closest_words :
        st.write(idx2word[index_word[1]]," -- ",index_word[0])

st.subheader("Mots les plus proches de...")
chosen_word = st.selectbox('Choix du mot', list(word2idx.keys())[:vocab_size])
nb_words = st.slider("Nb de mots",1,20,10)
print_closest(chosen_word,nb_words)

def print_compare(word1, word2, word3, number=10):
    result_list = compare(word2idx[word1], word2idx[word2],word2idx[word3],vectors, number)
    for index_word in result_list :
        st.write(idx2word[index_word[1]]," -- ",index_word[0])

st.subheader("Mot 1 - Mot 2 + Mot 3...")
compare_word1 = st.selectbox('Choix du mot 1', list(word2idx.keys())[:vocab_size],index =0)
compare_word2= st.selectbox('Choix du mot 2', list(word2idx.keys())[:vocab_size],index =1)
compare_word3 = st.selectbox('Choix du mot 3', list(word2idx.keys())[:vocab_size],index=2)
compare_words = st.slider("Nb mots",1,20,10)
print_compare(compare_word1,compare_word2,compare_word3,compare_words)