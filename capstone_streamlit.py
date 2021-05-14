

import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


# loading the trained model
pickle_in = open('./data/pickle_model.pkl', 'rb')
pickle_model = pickle.load(pickle_in)

pickle_vect = open('./data/pickle_vectorizer.pkl', 'rb')
pickle_vectorizer = pickle.load(pickle_vect)

#@st.cache()


st.title('MBTI Predictor')



st.text('Welcome to the Myers-Briggs Personality Type Predictor module. ')


txt = st.text_input('Please input text here:', value='')
# Preparing posts for model by vectorzing and filtering stop-words
if txt:

    vect = CountVectorizer(vocabulary=pickle_vectorizer.vocabulary_)
    X = vect.fit_transform([txt])

    predicted_personality = pickle_model.predict(X)[0]

#    st.write(f'Your personality type: {predicted_personality}')
    if predicted_personality == 0:
        st.write('INFJ')
    elif predicted_personality == 1:
        st.write('INFP')
    elif predicted_personality == 2:
        st.write('INTJ')
    else:
        st.write('INTP')
