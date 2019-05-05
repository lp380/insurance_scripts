from flask import Flask
from flask import request
import insurance_kmeans_three_cols as ins
app = Flask(__name__)

@app.route('/')
def hello_world():
    return "hello, world!"

@app.route('/kmeans')
def run_kmeans():
    return ins.from_external()

# accept arguemnts from url 
# http://127.0.0.1:3000/send-data?prescriptions=5&age=55&salary=65000
@app.route('/send-data')
def get_parameters():
    user_input_prescriptions = request.args.get('prescriptions', default = 1, type = int)
    user_input_age = request.args.get('age', default = 1, type = int)
    user_input_salary = request.args.get('salary', default = 1, type = int)
    var =  print_args(user_input_prescriptions, user_input_age, user_input_salary)
    return str(var)

def print_args(pres, age, salary):
    print("number of prescriptions from user", pres)
    print("age of user", age)
    print("salary of user", salary)
    return salary
