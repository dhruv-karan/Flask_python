from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators, IntegerField
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


import pickle
import os
import sqlite3
import numpy as np
# 
app = Flask(__name__)
cur_dir = os.path.dirname(__file__)
model = pickle.load(open("model.pkl","rb"))
clf = pickle.load(open(os.path.join(cur_dir,'model.pkl'),'rb'))
db = os.path.join(cur_dir,'reviews.sqlite')

def clean(review):
    review = re.sub(r'\W', ' ', str(data_x[i]))
    review = re.sub(r'[0-9]', ' ', review)
    review = review.lower()
    review = re.sub(r'^br$', ' ', review)
    review = re.sub(r'\s+br\s+',' ',review)
    review = re.sub(r'\s+[a-z]\s+', ' ',review)
    review = re.sub(r'^b\s+', '', review)
    review = re.sub(r'\s+', ' ', review)
    vectorizer = TfidfVectorizer(stop_words = stopwords.words('english'))
    return review = vectorizer.toarray()


def classifier(review):
    label = {0:"negative",1:"positive"}
    X = clean(review)
    y = clf.predict(X)
    return label[y]

def sqlite_entry(path,review,y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review,sentiment,date)"\
    "VALUES(?,?,DATETIME('now'))",(review,y))
    conn.commit()
    conn.close()




@app.route('/')
def index():
    return "We gonna buuild text analyiseireeiyfuer"

if __name__ == "__main__":
    # app.run(debug=True)
