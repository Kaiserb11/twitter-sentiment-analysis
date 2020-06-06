import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.feature_extraction.text import TfidfTransformer


file = "E:/github_desktop/twitter-sentiment-analysis/dataset/train.csv"
data = pd.read_csv(file, sep=',', encoding = 'utf8')

#shit load of preprocessing...

sent = data['original_text']
sent = [i.lower() for i in sent]
common_words = stopwords.words('english')
sent = pd.Series(sent).str.replace("[^a-zA-Z]", " ")
wrds = [i.split() for i in sent]
common_words = stopwords.words('english')
new_sentences = [[j for j in i if j not in common_words] for i in wrds]
vector = CountVectorizer()

x = [' '.join(i) for i in new_sentences]
y = data['sentiment_class'].tolist()

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

x_train = vector.fit_transform(x_train)
x_train = TfidfTransformer().fit_transform(x_train)
x_test = vector.transform(x_test)
x_test = TfidfTransformer().fit_transform(x_test)

clf = MultinomialNB()
clf.fit(x_train, y_train)

acc = clf.score(x_test, y_test)
print(acc)

