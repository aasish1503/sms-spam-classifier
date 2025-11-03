import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import os

# ✅ Ensure NLTK data is stored persistently
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.mkdir(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# ✅ Download required NLTK resources (only if missing)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_dir)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_dir)

# Load model and vectorizer
model = pickle.load(open('spam_classifier.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Preprocessing
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    text = nltk.word_tokenize(text)
    text = [ps.stem(word) for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    return ' '.join(text)

# Streamlit UI
st.title("📩 SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = vectorizer.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.error("❌ Spam Message")
    else:
        st.success("✅ Not a Spam Message")
