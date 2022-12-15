import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
glass=pd.read_csv("C:\\ex5\\glass.csv")
glass.head()
glass.tail() 
glass.shape
glass.isnull().sum()
sns.countplot(glass['Type'], color='red')
#Create a Naive Bayes object
nb = GaussianNB()
#Create variable x and y.
x = glass.drop(columns=['Type'])
y = glass['Type']
#Split data into training and testing data 
x_train, x_test, y_train, y_test = train_test_split(x, y, 
test_size=0.2, random_state=4)
#Training the model
nb.fit(x_train, y_train)
#Predict testing set
y_pred = nb.predict(x_test)
#Check performance of model
print(accuracy_score(y_test, y_pred))