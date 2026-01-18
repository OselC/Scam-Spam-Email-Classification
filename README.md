# Scam & Spam Email Classification App

A machine learning–based web application built with **Streamlit** to classify emails as **Not Spam (0)** or **Scam/Spam (1)** using Natural Language Processing (NLP) techniques.

Dataset: https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset
Streamlit App: https://scam-spam-email-classification-agggm7cm5vgx7fnh7jfn6o.streamlit.app/ 

---

## 🚀 Features

- Input any email text for real-time classification
- Automatic text preprocessing (tokenization, stopword removal, stemming)
- TF-IDF vectorization for feature extraction
- Multinomial Naive Bayes classifier
- Confidence visualization with a dual-color probability bar
- Ready for deployment on Streamlit Cloud

---

## 🧠 Machine Learning Pipeline

1. Text cleaning and normalization  
2. Tokenization using NLTK  
3. Stopword removal and stemming  
4. Feature extraction with TF-IDF  
5. Classification using Naive Bayes  
6. Probability-based confidence scoring  

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- Scikit-learn  
- NLTK  
- Pandas  
- NumPy  

---

## 📦 Installation & Usage

```bash
pip install -r requirements.txt
streamlit run app.py
```
