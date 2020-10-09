#import reqd libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#distance fucntion to be used in knn algorithm
def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))
#knn algorithm( x->training data, y->labels of training data,qp->query point, k->number of neighbours in consideration)    
def knn(x,y,qp,k=5):
    #vals list to store distances and their labels
    vals=[]
    #iterate over the training data and append distance and label for each point 
    for i in range(x.shape[0]):
        vals.append((dist(x[i],qp),y[i]))
    #sort the list so that k nearest can be taken
    vals=sorted(vals)
    #take first k nearest
    vals=vals[:k]
    #convert list into numpy array
    vals=np.array(vals)
    # now take how many unique labels are there in k nearest(use only 1st column for labels)
    #and also return their count
    #new vals is a list with two tupls
    #first tuple has labels and second has their freq.
    new_vals=np.unique(vals[:,1],return_counts=True)
    #take the index of max freq label
    max_freq_index=new_vals[1].argmax()
    #return the label with max freq
    return new_vals[0][max_freq_index]

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
	#check the shapes of data 
	#print(xtest.shape,ytest.shape)
	count=0
	# now test the knn for all values in xtest
	for i in range(xtest.shape[0]):
		ans=knn(xtrain,ytrain,xtest[i])
		print("actual label={} & predicted label={}".format(ytest[i],int(ans)))
		if ytest[i]!=ans:
			count+=1
	print("accuracy={}%".format((1-count/xtest.shape[0])*100))