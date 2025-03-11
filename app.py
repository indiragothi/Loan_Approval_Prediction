from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load('logistic_regression_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    gender = int(request.form.get('gender', 0))
    married = int(request.form.get('married', 0))
    dependents = int(request.form.get('dependents', 0))
    education = int(request.form.get('education', 0))
    self_employed = int(request.form.get('self_employed', 0))
    applicant_income = float(request.form.get('applicant_income', 0))
    coapplicant_income = float(request.form.get('coapplicant_income', 0))
    loan_amount = float(request.form.get('loan_amount', 0))
    term = int(request.form.get('term', 0))
    credit_history = int(request.form.get('credit_history', 0))
    area = int(request.form.get('area', 0))
    
    # Create feature array for prediction
    features = np.array([[gender, married, dependents, education, self_employed, 
                         applicant_income, coapplicant_income, loan_amount, 
                         term, credit_history, area]])
    
    # Make prediction
    prediction = model.predict(features)
    
    # Get prediction label
    result = "Approved" if prediction[0] == 1 else "Rejected"
    
    return render_template('index.html', prediction_text=f'Loan Application Status: {result}')

if __name__ == '__main__':
    app.run(debug=True)