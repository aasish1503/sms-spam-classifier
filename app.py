import streamlit as st
import pickle
import string
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model and vectorizer
model = pickle.load(open('spam_classifier.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Initialize stemmer
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    # Use regex split instead of nltk.word_tokenize
    words = re.findall(r'\b\w+\b', text)
    filtered = [ps.stem(word) for word in words if word not in stop_words and word not in string.punctuation]
    return ' '.join(filtered)

# UI
st.title("📩 SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() != "":
        transformed_sms = transform_text(input_sms)
        vector_input = vectorizer.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("❌ Spam Message")
        else:
            st.success("✅ Not a Spam Message")
    else:
        st.warning("Please enter a message to classify.")
