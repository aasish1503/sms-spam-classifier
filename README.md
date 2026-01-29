📩 SMS Spam Classifier using Machine Learning

🚀 Live Demo:
https://sms-spam-classifier-ikz9yutobvb8fajbongy2p.streamlit.app/

🧠 Project Overview

This project is an SMS Spam Classifier that predicts whether a given text message is Spam or Not Spam (Ham) using Natural Language Processing (NLP) and Machine Learning.

The model is deployed as an interactive web application using Streamlit, allowing users to enter an SMS message and instantly get predictions.

📊 Dataset Information

Dataset Name: SMS Spam Collection Dataset

Source: Kaggle / UCI Machine Learning Repository

Total Messages: 5,572

Classes:

ham → Not Spam

spam → Spam

⚙️ Tech Stack

Language: Python

Libraries: scikit-learn, nltk, pandas, streamlit

Model: Multinomial Naive Bayes

Vectorization: TF-IDF

🔍 How It Works

User enters an SMS message

Text preprocessing:

Lowercasing

Removing punctuation

Stopword removal

Stemming

Text is converted into numeric features using TF-IDF

Trained Naive Bayes model predicts Spam or Not Spam

Result is displayed instantly on the web interface

🧪 Sample Inputs & Outputs

Spam Example
Input:
Congratulations! You have won a free gift card. Click now!
Output:
Spam Message

Not Spam Example
Input:
Can you call me when you’re free?
Output:
Not a Spam Message

📂 Project Structure

sms-spam-classifier/
├── app.py
├── spam_classifier.pkl
├── vectorizer.pkl
├── requirements.txt
└── dataset/
└── spam.csv

▶️ Run Locally

pip install -r requirements.txt
streamlit run app.py

Then open:
http://localhost:8501

🌐 Deployment

Deployed on Streamlit Community Cloud

Automatically redeploys on GitHub commits

Public and shareable link

🎯 Key Highlights

End-to-end NLP and ML pipeline

High accuracy (~97–98%)

Lightweight and fast predictions

Fully deployed live web application

📌 Resume Description

Built an SMS Spam Classifier using NLP and Machine Learning. Implemented TF-IDF vectorization with Multinomial Naive Bayes and deployed the model as a live Streamlit web application for real-time spam detection.
