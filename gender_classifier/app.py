from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators, IntegerField

import pickle
import os
import sqlite3
import numpy as np

app = Flask(__name__)
cur_dir = os.path.dirname(__file__)
model = pickle.load(open("model.pkl","rb"))
clf = pickle.load(open(os.path.join(cur_dir,'model.pkl'),'rb'))

db = os.path.join(cur_dir,'reviews.sqlite')

def classifier(height,weight):
    label = {0:'female',1:'male'}
    X = np.array([[np.float64(height),np.float64(weight)]])
    y = clf.predict(X)
    if y < 0.5:
        y == 0
    else:
        y == 1
    return label[y]

def train(height,weight,y):
    X = np.array([[height,weight]])
    clf.partial_fit(X,y)

def sqlite_entry(path,height,weight,y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO review_db (height,weight,gender,date)"\
    "VALUES(?,?,?,DATETIME('now'))",(height,weight,y))
    conn.commit()
    conn.close()


#================================== routing or can be seen logic
class Dataform(Form):
    height = IntegerField('height', [validators.NumberRange(min=10, max=200)] )
    weight =  IntegerField('weight', [validators.NumberRange(min=10, max=300)] )
    

@app.route('/')
def index():
    form = Dataform(request.form)
    return render_template('prediction_form.html', form=form)


@app.route('/result',methods=['POST'])
def result():
    form  = Dataform(request.form)
    if request.method == 'POST' and form.validate():
        height = request.form['height']
        weight = request.form['weight']
        y = classifier(height,weight)
        return render_template('result.html',
        height=height,weight = weight,prediction=y)
    else:
        return render_template('prediction_form.html', form = form)
  
@app.route('/thanks',methods=['POST'])
def feedback():
    feedback = request.form('feedback_button')
    height = request.form['height']
    weight = request.form['weight']
    prediction = request.form['prediction']
    inv_label = {'female':0,'male':1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    train(height,weight,y)
    sqlite_entry(db,height,weight,y)
    return render_template('thanks.html')



if __name__ == "__main__":
    app.run(debug=True)
