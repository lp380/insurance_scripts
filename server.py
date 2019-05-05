from flask import Flask
import insurance_kmeans_three_cols as ins
app = Flask(__name__)

@app.route('/')
def hello_world():
    return "hello, world!"

@app.route('/kmeans')
def run_kmeans():
    return ins.from_external()



