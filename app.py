import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report
import joblib
import json
import os
from flask import Flask, flash, redirect, render_template, request, session, url_for


app = Flask(__name__)
app.secret_key = 'supersecretkey123'

# Home page
@app.route("/")
def home():
    return render_template("home.html")

# login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == 'admin' and password == 'admin':
            session['user_logged_in'] = True
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
    session.pop('user_logged_in', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


"""
This script trains multiple supervised learning models to predict telecom customer churn.  
It uses labeled data to evaluate and compare model performance.  
The goal is to identify customers likely to churn for proactive retention strategies.
"""

#pre_processing:
def Pre_processing():
    df = pd.read_csv("Telco-Customer-Churn.csv")

    # Handle TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Target encode
    target = LabelEncoder()
    df["Churn"] = target.fit_transform(df["Churn"])

    # Define feature groups
    num = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    nominal = ["Partner", "Dependents", "PhoneService", "PaperlessBilling",
               "gender", "MultipleLines", "InternetService", "OnlineSecurity",
               "OnlineBackup", "DeviceProtection", "TechSupport", 
               "StreamingTV", "StreamingMovies", "PaymentMethod"]
    ordinary = ["Contract"]

    # Label encoding
    label_encoding = {}
    for col in nominal:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoding[col] = le

    # Ordinal encode
    ord = OrdinalEncoder(categories=[["Month-to-month", "One year", "Two year"]])
    df[ordinary] = ord.fit_transform(df[ordinary])

    # Prepare X and y
    X = df.drop(["customerID", "Churn"], axis=1)
    y = df["Churn"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numeric features
    scaling = StandardScaler()
    X_train[num] = scaling.fit_transform(X_train[num])
    X_test[num] = scaling.transform(X_test[num])

    return X_train, X_test, y_train, y_test, label_encoding, ord, scaling, target

# Logistic Regression model :
def logistic_regression():
    
    X_train, X_test, y_train, y_test, label_encoding, ord, scaling, target = Pre_processing()

    model = LogisticRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    report = classification_report(y_test,y_pred)
    print(f"Logistic Regression Accuracy : {accuracy}")
    print(f"Logistic Regression Classification Report : \n {report}")
    model_filename = "Trained models/Logistic_Regression_model.pkl"
    joblib.dump(model,model_filename)
    return {"Model name": "Logistic Regression", "Accuracy" : accuracy, "Saved Model File":model_filename}
    
# KNN model :
def k_nearest_neighbour():
    
    X_train, X_test, y_train, y_test, label_encoding, ord, scaling, target = Pre_processing()

    model = KNeighborsClassifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    report = classification_report(y_test,y_pred)
    print(f"K Nearest Neighbour Accuracy : {accuracy}")
    print(f"K Nearest Neighbour Classification Report : \n{report}")
    model_filename = "Trained models/KNN_model.pkl"
    joblib.dump(model,model_filename)
    return {"Model name": "K Nearest Neighbour", "Accuracy" : accuracy, "Saved Model File":model_filename}

#Random Forest model:
def random_forest():

    X_train, X_test, y_train, y_test, label_encoding, ord, scaling, target = Pre_processing()

    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    report = classification_report(y_test,y_pred)
    print(f"Random Forest Accuracy : {accuracy}")
    print(f"Random Forest Classification Report : \n{report}")
    model_filename = "Trained models/RandomForest_model.pkl"
    joblib.dump(model,model_filename)
    return {"Model name": "Random Forest", "Accuracy" : accuracy, "Saved Model File":model_filename}

#SVC model:
def support_vector_classifier():

    X_train, X_test, y_train, y_test, label_encoding, ord, scaling, target = Pre_processing()

    model = SVC(kernel='rbf', gamma='scale', C=1.0)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    report = classification_report(y_test,y_pred)
    print(f"Support Vector Classifier Accuracy : {accuracy}")
    print(f"Support Vector Classifier Classification Report :\n {report}")
    model_filename = "Trained models/SVC_model.pkl"
    joblib.dump(model,model_filename)
    return {"Model name": "Support Vector Classifier", "Accuracy" : accuracy, "Saved Model File":model_filename}

#XGBoost model:
def xg_boost():

    X_train, X_test, y_train, y_test, label_encoding, ord, scaling, target = Pre_processing()

    model = XGBClassifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    report = classification_report(y_test,y_pred)
    print(f"XGBoost Classifier Accuracy : {accuracy}")
    print(f"XGBoost Classifier Classification Report : \n{report}")
    model_filename = "Trained models/XGBoost_model.pkl"
    joblib.dump(model,model_filename)
    return {"Model name": "XGBoost Classifier", "Accuracy" : accuracy, "Saved Model File":model_filename}


def model_for_prediction():

    logist = logistic_regression()
    print(logist)
    print("-----------------------------------------------------------------------------------------------------------")
    knn = k_nearest_neighbour()
    print(knn)
    print("-----------------------------------------------------------------------------------------------------------")
    random = random_forest()
    print(random)
    print("-----------------------------------------------------------------------------------------------------------")
    svm = support_vector_classifier()
    print(svm)
    print("-----------------------------------------------------------------------------------------------------------")
    xg = xg_boost()
    print(xg)
    print("-----------------------------------------------------------------------------------------------------------")
    
    all_models = [logist,knn,random,svm,xg]

    json_file_path = "Trained models/model_results.json"
    with open(json_file_path, "w") as json_file:
        json.dump(all_models, json_file, indent=4)

    with open(json_file_path, "r") as f:
        model_results = json.load(f)
    
    highest_accuracy_model = max(model_results, key=lambda x: x['Accuracy'])
    # Extract details
    model_type = highest_accuracy_model['Model name']
    accuracy = highest_accuracy_model['Accuracy']
    saved_model = highest_accuracy_model['Saved Model File']
    best_model = joblib.load(saved_model)

    return best_model

# model_for_prediction()

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if not session.get('user_logged_in'):
        flash('Please login first.', 'danger')
        return redirect(url_for('login'))

    prediction_result = None  # default

    if request.method == "POST":
        # Collect input from form
        customerID = request.form.get("customerID")
        gender = request.form.get("gender")
        SeniorCitizen = int(request.form.get("SeniorCitizen"))
        Partner = request.form.get("Partner")
        Dependents = request.form.get("Dependents")
        tenure = int(request.form.get("tenure"))
        PhoneService = request.form.get("PhoneService")
        MultipleLines = request.form.get("MultipleLines")
        InternetService = request.form.get("InternetService")
        OnlineSecurity = request.form.get("OnlineSecurity")
        OnlineBackup = request.form.get("OnlineBackup")
        DeviceProtection = request.form.get("DeviceProtection")
        TechSupport = request.form.get("TechSupport")
        StreamingTV = request.form.get("StreamingTV")
        StreamingMovies = request.form.get("StreamingMovies")
        Contract = request.form.get("Contract")
        PaperlessBilling = request.form.get("PaperlessBilling")
        PaymentMethod = request.form.get("PaymentMethod")
        MonthlyCharges = float(request.form.get("MonthlyCharges"))
        TotalCharges = float(request.form.get("TotalCharges"))

        # Create DataFrame from form data
        df = pd.DataFrame([{
            "customerID" :customerID,
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }])

        df = df.drop(["customerID"], axis= 1)
        model = model_for_prediction()

        # Define feature groups
        num = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
        nominal = ["Partner", "Dependents", "PhoneService", "PaperlessBilling",
                    "gender", "MultipleLines", "InternetService", "OnlineSecurity",
                    "OnlineBackup", "DeviceProtection", "TechSupport", 
                    "StreamingTV", "StreamingMovies", "PaymentMethod"]
        ordinary = ["Contract"]

        # Label encoding
        for col in nominal:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        # Ordinal encode
        ord = OrdinalEncoder(categories=[["Month-to-month", "One year", "Two year"]])
        df[ordinary] = ord.fit_transform(df[ordinary])

        # Scale numeric features
        scaling = StandardScaler()
        df[num] = scaling.fit_transform(df[num])

        # Predict churn (0 or 1)
        prediction = model.predict(df)[0]
        prediction_result = "Churn" if prediction == 1 else "Not Churn"

        # Add prediction to df
        df["Churn"] = prediction_result

        # Save or append to CSV
        csv_file = "Trained models/Predicted_Churn_Records.csv"
        file_exists = os.path.isfile(csv_file)
        df.to_csv(csv_file, mode='a', index=False, header=not file_exists)

    return render_template('dashboard.html', prediction_result=prediction_result)


if __name__ == '__main__':
    app.run(debug=True)

