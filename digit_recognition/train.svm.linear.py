#!/usr/bin/python2.7

from io import read_csv,write_delimited_file
from sklearn import decomposition
from sklearn import svm
import numpy as np

def main() :
    print "Reading training data"
    train = read_csv('train.csv',has_header=True)
    print "Reading testing data"
    test  = read_csv('test.csv',has_header=True)

    #the first column of the training set contains the training labels
    labels = [x[0] for x in train]
    train  = [x[1:] for x in train]

    #pca = decomposition.RandomizedPCA(n_components=150, whiten=True)
    #pca.fit(train)
    #train_pca = pca.transform(train)
    #test_pca  = pca.transform(test)

    clf = svm.LinearSVC()
    clf.fit(train,labels)
    output = clf.predict(test)
    output = [int(x) for x in output]

    f = open('prediction.svm.linear.csv','w')
    np.savetxt(f,output)
    f.close()


if __name__=="__main__":
    main()

