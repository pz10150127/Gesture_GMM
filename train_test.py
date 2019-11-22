from __future__ import division
import numpy as np
import csv
from numpy import *
from sklearn.mixture import GaussianMixture
import time

def load_modeling_samples():
    # Load the Modeling Samples
    csv_file = csv.reader(open('training.csv', 'r'))
    list = []
    L = []
    X_train = []
    for S in csv_file:
        S = map(eval, S)
        list = list + [S]
    for i in range(10):
        L.append(list[5000*i:(5000*(i+1)-1)])

    for i in range(10):
        X_train.append(np.array(L[i]))
    return X_train


def load_testing_samples():
    # Load the testing samples
    csv_file = csv.reader(open('testing.csv', 'r'))
    G = []
    list1 = []
    for k in csv_file:
        k = map(eval, k)
        list1 = list1 + [k]

    for i in range(10):
        G.append(list1[3000*i:(3000*(i+1)-1)])

    return G


def trainingGMMs(X_train):
    # Training the GMM models
    estimators = []
    for C in range(10):
        class_estimates = []
        for K in range(10):
            estimator =GaussianMixture(n_components=(K + 1), covariance_type="tied",max_iter=100)
            estimator.fit(X_train[C])

            class_estimates.append(estimator)
            if K==2:
                print 'AA'
                print  estimator.covariances_
                print 'BB'
                print  estimator.n_components
                print  'CC'
                print  estimator.means_
                print 'DD'
                print  estimator.weights_
        estimators.append(class_estimates)
    return estimators


if __name__ == '__main__':
    X_train = load_modeling_samples()
    X_star  = load_testing_samples()
    estimators = trainingGMMs(X_train)
    print "Training Done!"
    results=[]
    for kk in range(10):
       # kk=9
       # print kk
        sample_count=[]
        for i in range(10):
            true_count=0
            print "======%d"%i

            for j in range(len(X_star[i])):
                starttime = time.time()
                loglikelihood=[]
                for k in range(10):
                    #logprob, responsibilities = estimators[k][kk].score_samples(X_star[i][j])
                    logprob = estimators[k][kk].score(np.array(X_star[i][j]).reshape(-1,5))
                    loglikelihood.append(logprob)
                if i== loglikelihood.index(max(loglikelihood)):
                    true_count=true_count+1
                print (time.time()-starttime)
            sample_count.append(double(true_count)/len(X_star[i]))
        results.append(sample_count)

   # print results
    f = open("score.csv", "w")
    for kk in range(10):
        f.write("%s\n" % str(results[kk]))
