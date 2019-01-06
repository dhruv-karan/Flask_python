from flask import Flask, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
arr = np.array([[5,69]])
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
