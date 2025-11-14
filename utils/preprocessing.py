import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'\n+', ' ', str(text))
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    text = text.lower()
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)
