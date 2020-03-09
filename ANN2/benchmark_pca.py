import numpy as np
from util import get_normalized_data,get_transformed_data,y2indicator,cost,forward,gradb,gradW,error_rate
import matplotlib.pyplot as plt


def benchmark_full():
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

    print("Performing logistic regression...")
    # lr = LogisticRegression(solver='lbfgs')


    # convert Ytrain and Ytest to (N x K) matrices of indicator variables
    N, D = Xtrain.shape
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)
    LL = []
    LLtest = []
    CRtest = []

    # reg = 1
    # learning rate 0.0001 is too high, 0.00005 is also too high
    # 0.00003 / 2000 iterations => 0.363 error, -7630 cost
    # 0.00004 / 1000 iterations => 0.295 error, -7902 cost
    # 0.00004 / 2000 iterations => 0.321 error, -7528 cost

    # reg = 0.1, still around 0.31 error
    # reg = 0.01, still around 0.31 error
    lr = 0.00004
    reg = 0.01
    for i in range(500):
        p_y = forward(Xtrain, W, b)
        # print "p_y:", p_y
        ll = cost(p_y, Ytrain_ind)
        LL.append(ll)

        p_y_test = forward(Xtest, W, b)
        lltest = cost(p_y_test, Ytest_ind)
        LLtest.append(lltest)
        
        err = error_rate(p_y_test, Ytest)
        CRtest.append(err)

        W += lr*(gradW(Ytrain_ind, p_y, Xtrain) - reg*W)
        b += lr*(gradb(Ytrain_ind, p_y) - reg*b)
        if i % 10 == 0:
            print("Cost at iteration %d: %.6f" % (i, ll))
            print("Error rate:", err)

    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    iters = range(len(LL))
    plt.plot(iters, LL, iters, LLtest)
    plt.show()
    plt.plot(CRtest)
    plt.show()


def benchmark_pca():
    Xtrain, Xtest, Ytrain, Ytest = get_transformed_data()
    print("Performing logistic regression...")

    N, D = Xtrain.shape
    Ytrain_ind = np.zeros((N, 10))
    for i in range(N):
        Ytrain_ind[i, Ytrain[i]] = 1

    Ntest = len(Ytest)
    Ytest_ind = np.zeros((Ntest, 10))
    for i in range(Ntest):
        Ytest_ind[i, Ytest[i]] = 1

    W = np.random.randn(D, 10) / np.sqrt(D)
    b = np.zeros(10)
    LL = []
    LLtest = []
    CRtest = []

    # D = 300 -> error = 0.07
    lr = 0.0001
    reg = 0.01
    for i in range(200):
        p_y = forward(Xtrain, W, b)
        # print "p_y:", p_y
        ll = cost(p_y, Ytrain_ind)
        LL.append(ll)

        p_y_test = forward(Xtest, W, b)
        lltest = cost(p_y_test, Ytest_ind)
        LLtest.append(lltest)

        err = error_rate(p_y_test, Ytest)
        CRtest.append(err)

        W += lr*(gradW(Ytrain_ind, p_y, Xtrain) - reg*W)
        b += lr*(gradb(Ytrain_ind, p_y) - reg*b)
        if i % 10 == 0:
            print("Cost at iteration %d: %.6f" % (i, ll))
            print("Error rate:", err)

    p_y = forward(Xtest, W, b)
    print("Final error rate:", error_rate(p_y, Ytest))
    iters = range(len(LL))
    plt.plot(iters, LL, label = 'train loss')
    plt.plot(iters, LLtest, label='test loss')
    plt.title('Loss')
    plt.legend()
    plt.show()
    plt.plot(CRtest)
    plt.title('Error')
    plt.show()


if __name__ == '__main__':
    benchmark_pca()
    #benchmark_full()
