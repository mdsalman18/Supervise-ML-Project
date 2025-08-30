import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from xgboost import XGBClassifier

df = pd.read_csv("Telco-Customer-Churn.csv")

# Pre-processing

def Encode(features):
    label_encoding = {}
    for col in features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoding[col] =le
    return label_encoding

categoreis = df.drop(['customerID','SeniorCitizen','tenure','MonthlyCharges'], axis=1)

Encode(categoreis)

X =df.drop(['customerID','Churn'], axis=1)
y = df['Churn']

scaling = StandardScaler()
X = scaling.fit_transform(X)

X_tain,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = XGBClassifier()
model.fit(X_tain,y_train)

y_pred = model.predict(X_test)
Accuracy = accuracy_score(y_test,y_pred)
report = classification_report(y_test,y_pred)


print(f"Accuracy {Accuracy}")
print(f"report {report}")