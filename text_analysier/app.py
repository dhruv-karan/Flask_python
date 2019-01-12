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

@app.route('/')
def index():
    return "We gonna buuild text analyiseireeiyfuer"

if __name__ == "__main__":
    app.run(debug=True)
