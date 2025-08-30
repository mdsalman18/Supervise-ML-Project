import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report
import joblib

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
    model_filename = "Logistic_Regression_model.pkl"
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
    model_filename = "KNN_model.pkl"
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
    model_filename = "RandomForest_model.pkl"
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
    model_filename = "SVC_model.pkl"
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
    model_filename = "XGBoost_model.pkl"
    joblib.dump(model,model_filename)
    return {"Model name": "XGBoost Classifier", "Accuracy" : accuracy, "Saved Model File":model_filename}


def model_svainng():

    logistic = logistic_regression()
    print(logistic)
    knn = k_nearest_neighbour()
    print(knn)
    random = random_forest()
    print(random)
    svm = support_vector_classifier()
    print(svm)
    xg = xg_boost()
    print(xg)

    models = [logistic,knn,random,svm,xg]

    







