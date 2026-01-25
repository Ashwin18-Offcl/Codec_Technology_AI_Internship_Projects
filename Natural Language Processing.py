import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text corpus
documents = [
    "I love AI and machine learning.",
    "Natural Language Processing is fascinating."
]

# Tokenization
tokens = [word_tokenize(sentence) for sentence in documents]
print("Tokens:", tokens)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

print("TF-IDF Feature Names:", vectorizer.get_feature_names_out())
print("TF-IDF Vectors:\n", X.toarray())
