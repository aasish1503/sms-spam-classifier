import streamlit as st
import pickle
import string
import re
import nltk
from nltk.stem.porter import PorterStemmer

# Ensure stopwords are available
try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = pickle.load(open('spam_classifier.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Initialize stemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    words = re.findall(r'\b\w+\b', text)
    filtered = [ps.stem(word) for word in words if word not in stop_words and word not in string.punctuation]
    return ' '.join(filtered)

# Streamlit UI
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
