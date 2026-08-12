from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
import numpy as np 
import pandas as pd 
import  matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay , accuracy_score 

df = pd.read_csv('creditcard.csv') 
model = RandomForestClassifier(class_weight='balanced',random_state=42) 
scaler = StandardScaler() 
X = df.drop(['Class'],axis=1)
Y = df['Class']

X_train ,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2 , random_state=42)

X_train[['Time','Amount']] = scaler.fit_transform(X_train[['Time','Amount']])
X_test[['Time','Amount']] = scaler.transform(X_test[['Time','Amount']])

model.fit(X_train,Y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(Y_test, y_pred) 
print("the acc of this model is : "+str(round(acc*100,2))+"%")
conf = confusion_matrix(Y_test,y_pred)

disp=ConfusionMatrixDisplay(confusion_matrix=conf)
disp.plot(cmap='Reds')
plt.show() 
#f