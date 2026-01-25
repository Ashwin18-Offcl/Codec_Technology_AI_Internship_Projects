from textblob import TextBlob

text = "I love this product! It works perfectly."
blob = TextBlob(text)

# Senttiment polarity (-1 to 1)
print("Sentiment Polarity:", blob.sentiment.polarity)

# Sentiment classification
if blob.sentiment.polarity > 0:
    print("The sentiment is Positive.")
elif blob.sentiment.polarity < 0:
    print("The sentiment is Negative.")
else:
    print("The sentiment is Neutral.")
