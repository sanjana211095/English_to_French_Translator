import streamlit as st
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import os
# Load the pre-trained model
model = load_model("Minimodel.h5")

# Tokenizers
english_data = "small_vocab_en.txt"
french_data = "small_vocab_fr.txt"

def load_data(path):
    input_file = os.path.join(path)
    with open (input_file,"r") as f:
        data = f.read()
    return data.split('\n')

english_sentences = load_data(english_data)
french_sentences = load_data(french_data)

# def logits_to_text(logits,tokenizer):
#     index_to_words = {id: word for word, id in tokenizer.word_index.items()}

#     index_to_words[0] = '<PAD>'

#     return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits,1)])
def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    translated_words = []
    for prediction in np.argmax(logits, axis=1):
        word = index_to_words.get(prediction, None)
        if word is not None:
            translated_words.append(word)

    return ' '.join(translated_words)



def tokenize(x):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x),tokenizer

#define process function with x and y
def pad(x, length=None):
    return pad_sequences(x, maxlen=length, padding='post')
def preprocess(x,y):
    preprocess_x,x_tk = tokenize(x)
    preprocess_y,y_tk = tokenize(y)

    #padding the data
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    #keras's sparese_categorical_crossentropy function requires the labels to be in 3 dimension
    #Expanding dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape,1)
    return preprocess_x,preprocess_y,x_tk,y_tk

#preproc_english_sentence,preproc_french_sentence,english_tokenizer,french_tokenizer
preproc_english_sentences,preproc_french_sentences,english_tokenizer,french_tokenizer  =\
preprocess(english_sentences,french_sentences)
st.image("image.jpeg",width=500)
st.title("Language Translator(English to French)")

# Input text box for user input
st.warning("Please use lowercase only")
user_input = st.text_input("Enter an English sentence:")
if user_input:
    # Tokenize and pad the input
    sentence = [english_tokenizer.word_index.get(word, 0) for word in user_input.split()]
    sentence = pad_sequences([sentence], maxlen=preproc_french_sentences.shape[-2], padding='post')

    # Make predictions using the model
    predictions = model.predict(sentence[:1])[0]

    # Convert predictions to text
    translated_text = logits_to_text(predictions, french_tokenizer)

    # Remove padding from the generated text
    translated_text = translated_text.replace('<PAD>', '').strip()

    # Display the translated text
    st.subheader("Translated French Sentence:")
    st.write(translated_text)
