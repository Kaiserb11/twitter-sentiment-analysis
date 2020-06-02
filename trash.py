import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

txt = ['my name is Krishanu. I am a boy', 'My name is kriti. I am a girl']

sent = np.array(txt)

sent = [i.lower() for i in sent]

common_words = stopwords.words('english')

sent = pd.Series(sent).str.replace("[^a-zA-Z\d]", " ")

wrds = [i.split() for i in sent]
common_words = stopwords.words('english')

new_sentences = [[j for j in i if j not in common_words] for i in wrds]

print(new_sentences)

for i in new_sentences:
    for j in i:
        if i.count == 1:
            new_sentences.remove(i)
print(new_sentences)