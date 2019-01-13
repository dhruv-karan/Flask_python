from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
from sklearn.feature_extraction.text import TfidfVectorizer

# from nltk.corpus import stopwords
import re


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
    review = re.sub(r'\W', ' ', review)
    review = re.sub(r'[0-9]', ' ', review)
    review = review.lower()
    review = re.sub(r'^br$', ' ', review)
    review = re.sub(r'\s+br\s+',' ',review)
    review = re.sub(r'\s+[a-z]\s+', ' ',review)
    review = re.sub(r'^b\s+', '', review)
    review = re.sub(r'\s+', ' ', review)
    return review

def toArray(review):
    corpus = [review]
    # vectorizer = TfidfVectorizer(stop_words = stopwords.words('english'))
    vectorizer = TfidfVectorizer()
    review  = vectorizer.fit_transform(corpus).toarray()
    # review = np.reshape(review,(1,2000))
    # review = np.array[[vectorizer]]
    return review

def classifier(review):
    label = {0:"negative",1:"positive"}
    #review = clean(review)
    X = toArray(review)
    y = clf.predict(X)
    return label[y]

def sqlite_entry(path,review,y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review,sentiment,date)"\
    "VALUES(?,?,DATETIME('now'))",(review,y))
    conn.commit()
    conn.close()

#===================== routing nd pages=====================

class Reviewform(Form):
    review = TextAreaField('review')

@app.route('/')
def index():
    form = Reviewform(request.form)
    return render_template('prediction_form.html', form = form)

@app.route('/result',methods=['POST'])
def result():
    form = Reviewform(request.form)
    if request.method == 'POST':
        review = request.form['review']
        y = classifier(review)
        return render_template('result.html',
        review= review,prediction=y)
    else:
        return render_template('prediction_form.html', form = form)

# @app.route('./thanks',methods=['POST'])
# def feedback():
#     feedback = request.form('feedback_button')
#     review = request.form['review']
#     prediction = request.form['prediction']
#     inv_label = {'negative':0,'positive':1}
#     y = inv_label[prediction]
#     if feedback == 'Incorrect':
#         y = int(not(y))
#     # train(review,y)
#     sqlite_entry(db,review,y)
#     return render_template('thanks.html')



if __name__ == "__main__":
    app.run(debug=True)
