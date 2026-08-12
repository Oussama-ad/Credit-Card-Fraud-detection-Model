from imblearn.over_sampling import SMOTE 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as  plt
import numpy as np 
import pandas as pd 
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay , accuracy_score


df = pd.read_csv('creditcard.csv')
model_smote = RandomForestClassifier(random_state=42 , n_jobs=-1)

X= df.drop(columns=['Class'])
Y= df['Class'] 

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
scaler = StandardScaler()

X_train[['Time','Amount']] = scaler.fit_transform(X_train[['Time','Amount']])
X_test[['Time','Amount']] = scaler.transform(X_test[['Time','Amount']])

# the smote part  : (only on training data )
smote= SMOTE( sampling_strategy='auto',   # 'auto' = balance minority to match majority fully               
    random_state=42)
X_train_smote , Y_train_smote = smote.fit_resample(X_train,Y_train) # like this we balanced the data 

model_smote.fit(X_train_smote,Y_train_smote)
y_pred = model_smote.predict(X_test)
acc = accuracy_score(Y_test,y_pred)
print("accuracy = "+str(acc*100)+"%")
conf = confusion_matrix(Y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf) 
disp.plot(cmap='Reds')
plt.show()
