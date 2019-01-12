import pandas as pd
import os
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords

basepath = 'aclImdb'

labels = {'pos': 1, 'neg': 0}
df = pd.DataFrame()


for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r', encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)


df.columns = ['review', 'sentiment']
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('IMDb_Reviews.csv', index=False, encoding='utf-8')

data = pd.read_csv('IMDB_Reviews.csv')
data_x = data.review
data_y = data.sentiment
data.columns
data.isnull().sum()

data.head(1)

corpus = []

for i in range(len(data_x)):
    review = re.sub(r'\W', ' ', str(data_x[i]))
    review = re.sub(r'[0-9]', ' ', review)
    review = review.lower()
    review = re.sub(r'^br$', ' ', review)
    review = re.sub(r'\s+br\s+',' ',review)
    review = re.sub(r'\s+[a-z]\s+', ' ',review)
    review = re.sub(r'^b\s+', '', review)
    review = re.sub(r'\s+', ' ', review)
    corpus.append(review)


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, data_y, test_size = 0.20, random_state = 0)


#from sklearn.ensemble import RandomForestRegressor
#regressor = RandomForestRegressor(n_estimators =2, random_state=0)
#regressor.fit(X_train,y_train)

#regressor.score(X_test,y_test)


# Training the classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)


# Testing model performance
pred = classifier.predict(X_test)

#accuracy
classifier.score(X_test,y_test)

#pickling model
pickle.dump(classifier, open("model.pkl","wb"))












