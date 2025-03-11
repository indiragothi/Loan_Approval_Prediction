# Loan Approval Prediction

## 📌 Introduction
Predicting loan approval is a crucial application of machine learning in the financial sector. Banks and financial institutions evaluate various factors before granting a loan. This project aims to build a **Loan Approval Prediction** system using **Machine Learning** to automate and improve the decision-making process.

## 🚀 Features
- Data Collection & Preprocessing
- Exploratory Data Analysis (EDA)
- Model Training & Evaluation
- Hyperparameter Tuning
- Model Deployment (Flask Integration Ready)

---

## 📊 Dataset
The dataset consists of historical loan applications, including:
- **Numerical Features:** Applicant Income, Coapplicant Income, Loan Amount, Loan Term, Credit History, etc.
- **Categorical Features:** Gender, Marital Status, Dependents, Education, Self-Employed, Property Area, etc.
- **Target Variable:** Loan Status (Approved/Not Approved)

---

## 🛠️ Steps Followed

### 1️⃣ Data Collection & Preprocessing
- Collected historical loan data.
- Handled missing values.
- Encoded categorical variables.
- Scaled numerical features using `StandardScaler`.

### 2️⃣ Exploratory Data Analysis (EDA)
- Checked the structure of the dataset.
- Visualized target variable distribution using **bar plots & pie charts**.
- Analyzed feature distributions with **histograms & box plots**.
- Identified and handled outliers.

### 3️⃣ Data Splitting
- Split the dataset into **training (70%)** and **testing (30%)** sets using `train_test_split`.

### 4️⃣ Model Selection
Implemented and compared the following models:
- ✅ **Logistic Regression**
- ✅ **Random Forest Classifier**
- ✅ **Support Vector Machine (SVM)**
- ✅ **XGBoost Classifier**

### 5️⃣ Model Training & Evaluation
- Trained models on the **training set**.
- Evaluated models using **accuracy, precision, recall, and F1-score**.

### 6️⃣ Hyperparameter Tuning
- Applied **Grid Search** for tuning hyperparameters.
- Improved model accuracy using the best-selected parameters.

### 🎯 Results
- The **best-performing model** was **Logistic Regression**, achieving an accuracy of **86%**.
- Further improvements can be made through **feature engineering** and **trying different models**.

---

## 💾 Model Deployment
To use the trained model in a web application, we saved the best model using `joblib`:
```python
import joblib
joblib.dump(log_reg_best, 'logistic_regression_model.pkl')
```
This model can be loaded in a **Flask API** for real-time predictions.

---

## 📌 How to Run the Project
### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 2️⃣ Train the Model
Run the Jupyter Notebook `loan-approval.ipynb` to train and save the model.

### 3️⃣ Run Flask App
```bash
python app.py
```
 
---

## 📌 Future Enhancements
- Feature engineering to improve accuracy.
- Deploying the model using **Streamlit** or **FastAPI**.
- Integrating a frontend UI for better user experience.

---

## 📢 Conclusion
This project demonstrates how **machine learning** can streamline loan approval processes, reducing manual effort while improving decision-making accuracy. 🚀

---
 

