from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
import numpy as np 
import pandas as pd 
import  matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay , accuracy_score ,average_precision_score
from  xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE 

df = pd.read_csv('creditcard.csv')  
scaler = StandardScaler() 
X = df.drop(['Class'],axis=1)
Y = df['Class']

X_train ,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2 , random_state=42)


X_train[['Time','Amount']] = scaler.fit_transform(X_train[['Time','Amount']])
X_test[['Time','Amount']] = scaler.transform(X_test[['Time','Amount']])

# the smote part  : (only on training data )
smote= SMOTE( sampling_strategy='auto',   # 'auto' = balance minority to match majority fully               
    random_state=42)
X_train_smote , Y_train_smote = smote.fit_resample(X_train,Y_train) # like this we balanced the data 


model_xgb_smote = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='aucpr')
model_xgb_smote.fit(X_train_smote, Y_train_smote)
y_pred = model_xgb_smote.predict(X_test)
y_proba = model_xgb_smote.predict_proba(X_test)[:, 1]

conf = confusion_matrix(Y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf)
disp.plot(cmap='Reds')
plt.title("XGBoost + SMOTE")
plt.show()

auprc = average_precision_score(Y_test, y_proba)
print(f"AUPRC: {round(auprc, 4)}")