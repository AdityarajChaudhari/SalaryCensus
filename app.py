import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle
import psycopg2
from flask_cors import CORS, cross_origin

app = Flask(__name__)
model = pickle.load(open('./ModelSaving/dtc.pkl', 'rb'))
print("Inside model")
scalar = pickle.load(open('./Scaler/Scalar.pkl', 'rb'))
print("inside scalar")

db = psycopg2.connect(host="ec2-44-199-86-61.compute-1.amazonaws.com",user="ucripultcfnyfq",password="8a83eb807254ce8f5e2b7c1281c7ab0b928c34ac4409c396f40f0dcfa7cd679a",database="d7qjn8oe2dejqu")
cur = db.cursor()
cur.execute("create table if not exists incomecensus(Age int, FinalWeight int, MaxEducationalQualification varchar(20), CapitalGain varchar(5), CapitalLoss varchar(5), WorkSector varchar(20), MaritalStatus varchar(15),"
            "Race varchar(15), Gender varchar(15), WorkStyle varchar(20), Country varchar(10) ,Income Varchar(30)  )")
db.commit()

@cross_origin()
@app.route('/', methods=['GET'])
def home():
    print("Inside home page")
    return render_template('./home.html')

@cross_origin()
@app.route('/info', methods=['GET'])
def info():
    print("Inside info page")
    return render_template('./info.html')

@cross_origin()
@app.route('/developer', methods=['GET'])
def developer():
    print("Inside home page")
    return render_template('./developer.html')

@cross_origin()
@app.route('/contact', methods=['GET'])
def contact():
    print("Inside contact page")
    return render_template('./contact.html')

@cross_origin()
@app.route('/app', methods=['GET'])
def index_page():
    print("Inside app")
    return render_template('./index.html')

@cross_origin()
@app.route('/predict', methods=['POST','GET'])
def predict():

    if request.method == 'POST':
        age = int(request.form['age'])

        wt = int(request.form['Final Weight'])

        edu = (request.form['Education'])
        if edu == 'Higher Studies':
            edu = 0
        elif edu == 'Bachelors':
            edu = 1
        elif edu == 'Associate':
            edu = 2
        elif edu == 'Prof-School':
            edu = 3
        elif edu == 'Diploma':
            edu = 4
        else:
            edu = 5

        gain = (request.form['CapitalGain'])
        if gain == 'Yes':
            gain = 1
        else:
            gain = 0

        loss = (request.form['CapitalLoss'])
        if loss == 'Yes':
            loss = 0
        else:
            loss = 1

        wrk_cls = (request.form['WorkClass'])
        if wrk_cls == 'Private':
            wrk_cls = 0, 0, 0
        elif wrk_cls == 'Government':
            wrk_cls = 0, 0, 1
        elif wrk_cls == 'SelfEmployeed':
            wrk_cls = 0, 1, 0
        else:
            wrk_cls = 1, 0, 0

        status = (request.form['MaritalStatus'])
        if status == 'Married':
            status = 0
        else:
            status = 1

        race = (request.form['race'])
        if race == 'White':
            race = 0, 0
        elif race == 'Brown':
            race = 1, 0
        else:
            race = 0, 1

        gen = (request.form['gender'])
        if gen == 'Male':
            gen = 1
        else:
            gen = 0

        hours = (request.form['hours'])
        if hours == 'ideal':
            hours = 0, 0
        elif hours == 'over':
            hours = 1, 0
        else:
            hours = 0, 1

        country = (request.form['country'])

        if country == 'US':
            country = 1
        else:
            country = 0

        col = ([[age, wt, edu, gain, loss, *wrk_cls, status, *race, gen, *hours, country]])
        #col = pd.DataFrame(col)
        print(col)
        scaled_col = scalar.transform(col)
        print(scaled_col)
        prediction = model.predict(scaled_col)
        print(prediction)

        col1 = int(request.form['age'])
        col2 = int(request.form['Final Weight'])
        col3 = (request.form['Education'])
        col4 = (request.form['CapitalGain'])
        col5 = (request.form['CapitalLoss'])
        col6 = (request.form['WorkClass'])
        col7 = (request.form['MaritalStatus'])
        col8 = (request.form['race'])
        col9 = (request.form['gender'])
        col10 = (request.form['hours'])
        col11 = (request.form['country'])


        if prediction == np.array(1):
            a = "More Than 50K"
            cur.execute(f"insert into incomecensus values{(col1, col2, col3, col4, col5, col6, col7, col8, col9 , col10, col11,a)}")
            db.commit()
            return render_template('./result.html', Prediction_text = "The Salary of an Individual is More than 50K")
        else:
            b = "Less Than 50K"
            cur.execute(f"insert into incomecensus values{(col1, col2, col3, col4, col5, col6, col7, col8, col9 , col10,col11,b)}")
            db.commit()
            return render_template('./result.html', Prediction_text = "The Salary of an Individual is Less than 50K")
    else:
        return render_template('./home.html')


if __name__ == "__main__":
    app.run(debug=True)