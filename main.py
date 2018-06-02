import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import csv

def titanic_pred():

    df = pd.read_csv("train.csv")  # reading the test data set

    df['Age'].replace('',28, inplace = True) # replacing all nan values with an avg age 28
    df.dropna(subset=['Age'], inplace = True)

    df['Sex'].replace('male', 0 , inplace = True) # giving male and female a binary digit for model to train
    df['Sex'].replace('female', 1 , inplace = True)

    y = df['Survived'].values
    X = df[list(['Pclass','Age','SibSp','Parch','Fare'])].values

    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 0)

    grd =  GradientBoostingClassifier().fit(X_train, y_train) # training the model

    df2 = pd.read_csv("test.csv") #reading the test set
    
    
    df2['Age'].replace('',28, inplace = True) # replacing all nan values with an avg age 28
    df2.dropna(subset=['Age'], inplace = True)
    for column in list(df2):
      df2[column] = df2[column].replace(r'\s+', np.nan, regex=True)
      df2[column] = df2[column].fillna(0)
    df2['Sex'].replace('male', 0 , inplace = True)
    df2['Sex'].replace('female', 1 , inplace = True)
    
    
    X_test = df2[list(['Pclass','Age','SibSp','Parch','Fare'])].values

    
    pred = grd.predict(X_test)   # predicting values on test dataset
    

    id = df2['PassengerId'].values
    ans = np.column_stack((id,pred)) # stacking passengerId and survived predictions in one array

    with open('ans.csv','w') as csvfile:
        csvfile.write('PassengerId,' + 'Survived\n')  # writing the vslues into csv file for submission

        for a in ans:
            for col in a:
                csvfile.write('%d, ' % col)
            csvfile.write("\n")

        
        csvfile.close()

    

    
titanic_pred()
