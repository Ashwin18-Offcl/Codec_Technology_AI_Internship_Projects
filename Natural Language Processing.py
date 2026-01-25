import nltk
nltk.download('punkt')

import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

#Sample text
text =["I love AI and machine learning.","Natural Language Processing is fascinating."]

#Tokenization
tokens = [word_tokenize(sentence) for sentence in text]
print("Tokens:", tokens)

#TF-IDF Vecrtorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text)
print("TF-IDF Feature Names:", vectorizer.get_feature_names_out())
print("TF-IDF Vectors:\n", X.toarray())