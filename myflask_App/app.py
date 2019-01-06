from flask import Flask

app = Flask(__name___)

app.route('/')
def index():
    return 'jdhgsfhgd'

if __name__ == "__main__":
    app.run(debug=True)
