from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators

import pickle
import os
import sqlite3 
import numpy as np

app = Flask(__name__)
cur_dir = os.path.dirname(__file__)
model = pickle.load(open("model.pkl","rb"))
clf = pickle.load(open(os.path.join(cur_dir,'model.pkl'),'rb'))

db = os.path.join(cur_dir,'reviews.sqlite')


def classifier(document):
    label = {0:'female',1:'male'}
    X = np.array((document))
    y = clf.predict(X)[0]
    if y <0.5:
        y ==0
    else:
        y ==1
    return label[y]
def train(document,y):
    X = np.array([[document]])
    clf.partial_fit(X,y)

def sqlite_entry(path,document,y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (review,sentiment,date)"\
    "VALUES(?,?,DATETIME('now'))",(document,y))
    conn.commit()
    conn.close()


#================================== routing or can be seen logic 
@app.route('/')
def index():
    return render_template('home.html')
    


@app.route('/predict')
def predict():
     if(model.predict(arr >0.5)):
         return 'male'
     else:
         return 'femlae'



if __name__ == "__main__":
    app.run(debug=True)
