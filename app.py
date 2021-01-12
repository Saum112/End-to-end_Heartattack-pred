# Importing essential libraries
from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the CLassifier model
filename = 'heartattack-prediction-model.pkl'
classifier = joblib.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/',methods=['GET'])
def home():
	return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        ex = int(request.form['exang'])
        old = float(request.form['oldpeak'])
        cp = int(request.form['cp'])
        thal= int(request.form['thalach'])
        sex = int(request.form['sex'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        age = int(request.form['age'])
        trestbps= int(request.form['trestbps'])
        restecg = int(request.form['restecg'])
        
        data = np.array([[ex,old, cp,thal, sex, chol, fbs,age,trestbps,restecg]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)