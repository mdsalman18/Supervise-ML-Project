# Telco Customer Churn Prediction Web App

This project is a **Telecom Customer Churn Prediction** web application built with **Flask** and multiple machine learning models. It allows users to input customer information and predicts whether the customer is likely to churn or stay. The project demonstrates end-to-end ML lifecycle from data preprocessing, model training, evaluation, to deployment via a web interface.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Dataset](#dataset)  
- [Technologies](#technologies)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Model Training](#model-training)  
- [Folder Structure](#folder-structure)  
- [Notes & Future Improvements](#notes--future-improvements)  
- [License](#license)

---

## Project Overview

Customer churn prediction helps telecom companies identify customers who are likely to leave their service. By predicting churn early, companies can proactively take measures to retain customers, reducing loss and improving revenue.

This project uses the **Telco Customer Churn** dataset from Kaggle to train and evaluate several supervised learning models — Logistic Regression, K-Nearest Neighbors, Random Forest, Support Vector Classifier, and XGBoost. The best-performing model is saved and deployed in a Flask web app, allowing real-time churn prediction based on user input.

---

## Features

- Data preprocessing including handling missing values and encoding categorical variables  
- Multiple classification models with accuracy and classification report output  
- Saves trained models for later inference  
- Flask web app with login/logout functionality  
- Input form with fields matching the Telco dataset features  
- Real-time churn prediction (Churn / Not Churn)  
- Stores prediction records in a CSV file for reference

---

## Dataset

The project uses the publicly available **Telco Customer Churn** dataset, which contains:

- Customer demographics (gender, senior citizen, partner, dependents)  
- Account information (tenure, contract type, payment method)  
- Service usage (internet, phone services, streaming)  
- Charges (monthly charges, total charges)  
- Churn label (Yes/No)

Dataset source: [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)

---

## Technologies

- Python 3.x  
- Flask  
- pandas  
- scikit-learn  
- xgboost  
- joblib  
- HTML/CSS (for frontend)

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/telco-customer-churn.git
   cd telco-customer-churn
Sure! Here’s a complete **README.md** template tailored for your Telco Customer Churn Prediction project using Flask and multiple ML models. It covers:

* Project overview
* Features
* Dataset info
* Installation and setup
* Usage instructions
* Model training explanation
* Folder structure
* Notes & improvements

---

````markdown
# Telco Customer Churn Prediction Web App

This project is a **Telecom Customer Churn Prediction** web application built with **Flask** and multiple machine learning models. It allows users to input customer information and predicts whether the customer is likely to churn or stay. The project demonstrates end-to-end ML lifecycle from data preprocessing, model training, evaluation, to deployment via a web interface.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Dataset](#dataset)  
- [Technologies](#technologies)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Model Training](#model-training)  
- [Folder Structure](#folder-structure)  
- [Notes & Future Improvements](#notes--future-improvements)  
- [License](#license)

---

## Project Overview

Customer churn prediction helps telecom companies identify customers who are likely to leave their service. By predicting churn early, companies can proactively take measures to retain customers, reducing loss and improving revenue.

This project uses the **Telco Customer Churn** dataset from Kaggle to train and evaluate several supervised learning models — Logistic Regression, K-Nearest Neighbors, Random Forest, Support Vector Classifier, and XGBoost. The best-performing model is saved and deployed in a Flask web app, allowing real-time churn prediction based on user input.

---

## Features

- Data preprocessing including handling missing values and encoding categorical variables  
- Multiple classification models with accuracy and classification report output  
- Saves trained models for later inference  
- Flask web app with login/logout functionality  
- Input form with fields matching the Telco dataset features  
- Real-time churn prediction (Churn / Not Churn)  
- Stores prediction records in a CSV file for reference

---

## Dataset

The project uses the publicly available **Telco Customer Churn** dataset, which contains:

- Customer demographics (gender, senior citizen, partner, dependents)  
- Account information (tenure, contract type, payment method)  
- Service usage (internet, phone services, streaming)  
- Charges (monthly charges, total charges)  
- Churn label (Yes/No)

Dataset source: [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)

---

## Technologies

- Python 3.x  
- Flask  
- pandas  
- scikit-learn  
- xgboost  
- joblib  
- HTML/CSS (for frontend)

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/telco-customer-churn.git
   cd telco-customer-churn
````

2. Create a virtual environment and activate it (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the `Telco-Customer-Churn.csv` dataset and place it in the project root directory.

---

## Usage

1. Train models and prepare encoders/scalers by running:

   ```bash
   python app.py
   ```

   This will train multiple models and save the best performing one.

2. Run the Flask app:

   ```bash
   flask run
   ```

3. Open your browser at `http://127.0.0.1:5000/`

4. Login using the default admin credentials:

   ```
   Username: admin
   Password: admin
   ```

5. Navigate to the dashboard, fill in the customer details, and get churn prediction results.

---

## Model Training

* Data preprocessing handles missing values (`TotalCharges`), encodes categorical variables using LabelEncoder and OrdinalEncoder, and scales numeric features.
* Multiple models are trained and evaluated using accuracy and classification reports.
* The best performing model is saved to disk using joblib for deployment.
* The app loads the saved model for prediction instead of retraining every time.

---

## Folder Structure

```
telco-customer-churn/
│
├── Trained models/           # Saved models and preprocessing objects
│   ├── Logistic_Regression_model.pkl
│   ├── KNN_model.pkl
│   ├── RandomForest_model.pkl
│   ├── SVC_model.pkl
│   ├── XGBoost_model.pkl
│   └── model_results.json
│
├── templates/                # HTML templates for Flask
│   ├── home.html
│   ├── login.html
│   └── dashboard.html
│
├── static/                   # Static files (CSS, JS, images)
│
├── Telco-Customer-Churn.csv # Dataset file
│
├── app.py                   # Main Flask app and model training code
├── requirements.txt         # Required Python packages
└── README.md                # Project documentation
```

---


**Feel free to contribute, report issues, or suggest enhancements!**

