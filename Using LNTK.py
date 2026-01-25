import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

text = "Apple is looking at buying U.K. startup for $1 billion."

# Sentence and word tokenization
sentences = sent_tokenize(text)
words = word_tokenize(text)
print("Sentences:", sentences)
print("Words:", words)

# Remove stop words
stop_words = set(stopwords.words('english'))
filtered_words = [w for w in words if w.lower() not in stop_words]
print("Filtered Words:", filtered_words)

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(w) for w in filtered_words]
print("Stemmed Words:", stemmed_words)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(w) for w in filtered_words]
print("Lemmatized Words:", lemmatized_words)

# Part-of-speech tagging
pos_tags = nltk.pos_tag(words)
print("POS Tags:", pos_tags)
