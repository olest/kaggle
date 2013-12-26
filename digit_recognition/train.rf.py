#!/usr/bin/python2.7

from digit.io import read_csv,write_delimited_file
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def main() :
    print "Reading training data"
    train = read_csv('train.csv',has_header=True)
    print "Reading testing data"
    test  = read_csv('test.csv',has_header=True)

    #the first column of the training set contains the training labels
    labels = [x[0] for x in train]
    train  = [x[1:] for x in train]

    rf = RandomForestClassifier(n_estimators=1000, n_jobs=1)
    rf.fit(train, labels)
    output = rf.predict(test)
    output = [int(x) for x in output];

    f = open('prediction.randomforest.n1000.csv','w')
    np.savetxt(f,output)
    f.close()


if __name__=="__main__":
    main()

