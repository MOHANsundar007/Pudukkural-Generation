import numpy as np
from flask import Flask, request, render_template
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
model = "gemini-pro"
llm = ChatGoogleGenerativeAI(model=model)
thirukural_model = load_model('thirukural_model.h5')


with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

def get_meaning(verse_text):
    user_message = f"இந்த புதுக்குறளின் எளிமையான அர்த்தத்தை கூறவும்: {verse_text}"
    response = llm.invoke([{"role": "user", "content": user_message}])
    meaning = response.content if hasattr(response, 'content') else 'No meaning found'
    return meaning

def generate_thirukural(seed_text, max_sequence_len, tokenizer, model, n_words=7):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    generated_text = seed_text

    for _ in range(n_words):
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted_probs, axis=-1)[0]
        predicted_word = tokenizer.index_word[predicted_word_index]

        generated_text += ' ' + predicted_word
        token_list = np.append(token_list, predicted_word_index)[1:]

    return generated_text


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/generate_and_get_meaning', methods=['POST'])
def generate_and_get_meaning():
    seed_text = request.form['seed_text']
    max_sequence_len = 15
    generated_thirukural = generate_thirukural(seed_text, max_sequence_len, tokenizer, thirukural_model, n_words=7)
    meaning = get_meaning(generated_thirukural)
    return render_template('result.html', seed_text=seed_text, generated_thirukural=generated_thirukural,
                           meaning=meaning)

if __name__ == '__main__':
    app.run(debug=True)
