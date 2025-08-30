import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

df = pd.read_csv("Telco-Customer-Churn.csv")

def Encode(Features):
    lebal_encoding= {}
    for col in Features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        lebal_encoding[col] = le
    return lebal_encoding

# Dataset loading
df = pd.read_csv("Telco-Customer-Churn.csv")

# Pre-Processing

categories = df.drop(['customerID','SeniorCitizen','tenure','MonthlyCharges'], axis=1)

Encode(categories)
    
X = df.drop(['Churn','customerID'], axis=1) 
y = df['Churn']

#scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train,y_train)

y_pred =model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
confusionMartix = confusion_matrix(y_test,y_pred)
print("Accuracy: \n",accuracy)
print("Confusion Matrix: \n",confusionMartix)
