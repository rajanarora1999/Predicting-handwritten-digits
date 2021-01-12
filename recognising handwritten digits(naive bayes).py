#import reqd libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
if __name__=='__main__':
	#make a pandas dataframe from set
    print("This model uses mnist dataset for recognising handwritten digits.")
    print("This dataset has 42000 samples as training data.")
    df=pd.read_csv('train.csv')

	#checked its shape
	#print(df.shape) i.e (42000,785)
	# check headers of columns( first is label and rest 784 columns are 28x28 pixel values) 
	#df.columns
	#convert into numpy array so that slicing can be done
    x=np.array(df)

	#print(x)
	#extract out labels(1st column)
    labels=x[:,0]
	#extract out pixel data
    x=x[:,1:]
	# split the  data into testing and training part
    print(" enter the percentage of data to be used as testing data=")
    n=int(input())
    k=1-((n%100)/100)
	#split the number of rows into training data and testing data
    split=int(k*x.shape[0])
    xtrain=x[:split,:]
    ytrain=labels[:split]
    xtest=x[split:,:]
    ytest=labels[split:]
    mnb=MultinomialNB()
    mnb.fit(xtrain,ytrain)
    preds=mnb.predict(xtest)
    count=0
    for i in range(xtest.shape[0]):
        print("actual label={} & predicted label={}".format(ytest[i],preds[i]))
    print("accuracy={}%".format(mnb.score(xtrain,ytrain)*100))