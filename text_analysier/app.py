from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return "We gonna buuild text analyiseireeiyfuer"

if __name__ == "__main__":
    app.run(debug=True)
