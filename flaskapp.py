import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('heartm.pkl', 'rb'))

def getParameters():
    parameters = []
    parameters.append(request.form['age'])
    parameters.append(request.form['sex'])
    parameters.append(request.form['cp'])
    parameters.append(request.form['trestbps'])
    parameters.append(request.form['chol'])
    parameters.append(request.form['fbs'])
    parameters.append(request.form['restecg'])
    parameters.append(request.form['thalach'])
    parameters.append(request.form['exang'])
    parameters.append(request.form['oldpeak'])
    parameters.append(request.form['slope'])
    parameters.append(request.form['ca'])
    parameters.append(request.form['thal'])
    return parameters


@app.route("/")
def home():
    return render_template('form.html')


@app.route('/predict',methods=['GET','POST'])
def predict():
    parameters = getParameters()
    inputFeature = np.asarray(parameters).reshape(1, -1)
    my_prediction = model.predict(inputFeature)



    print(inputFeature)
    print(my_prediction)

    output = round(float(my_prediction[0]), 2)
    if(output == 1):
        return render_template('form.html',prediction_text='High chances of heart disease.Consult doctor!')
    if (output == 0):
        return render_template('form.html', prediction_text='Low chances of heart disease. Chill !!')

if __name__ == "__main__":
    app.run(debug=True)