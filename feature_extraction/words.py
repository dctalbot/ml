from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

stemmer = SnowballStemmer("english")
vectorizer = CountVectorizer()
sw = stopwords.words("english")

s1 = "This is a sentence"
s2 = "This is another sentence"
s3 = "This is a third sentence and this is the last of the sentences"
bag = vectorizer.fit([s1, s2, s3])
bag = vectorizer.transform([s1, s2, s3])
# print(bag)

words = s1.split() + s2.split() + s3.split()
words = [stemmer.stem(w) for w in words]
words = list(set(words))
words = [w for w in words if w not in sw]
print(words)
