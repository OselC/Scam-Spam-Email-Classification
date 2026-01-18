import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK resources (only first time)
nltk.download('punkt')
nltk.download('stopwords')

# Load stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Preprocessing
def preprocess_email(message):
    if not isinstance(message, str):
        return ""

    message = ' '.join(message.split())
    words = word_tokenize(message)
    words = [w.lower() for w in words if w.isalpha() and w not in stop_words]
    words = [stemmer.stem(w) for w in words]

    return ' '.join(words)

# Load trained model
@st.cache_resource
def load_model():
    with open("model.pickle", "rb") as f:
        model, tfidf, _ = pickle.load(f)
    return model, tfidf

model, tfidf = load_model()

# Streamlit UI
st.title("📧 Spam / Scam Email Detection")

email_text = st.text_area("Enter email content:", height=200)

if st.button("Predict"):
    processed = preprocess_email(email_text)
    vectorized = tfidf.transform([processed])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]

    if prediction == 0:
        st.success("✅ This email is **NOT SPAM**")
    else:
        st.error("🚨 This email is **SCAM / SPAM**")

    predicted_confidence = probability[prediction] * 100
    st.markdown("### Confidence Score")
    st.write(f"{predicted_confidence:.2f}%")
    st.progress(predicted_confidence / 100)
