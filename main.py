import numpy as np

def OR_gate(x1,x2):
    return min(1,x1+x2)

def activationFunction(x):
    if x >= 0:
        return 1
    return 0

def zeroGrad(grad):
    for i in range(len(grad)):
        grad[i] = 0.0

def trainPerceptronWeights(X, Y, lr=0.1):
    converged = False
    W = np.array([0.0,0.0,0.0])
    grad = np.array([0.0,0.0,0.0])
    y_pred = np.zeros(len(X))
    itr_num=0
    max_itr=50

    while not converged:
        itr_num+=1
        product = X@W

        for i in range(len(X)):
            y_pred[i] = activationFunction(product[i])

        zeroGrad(grad)
        for i in range(len(W)):
            for j in range(len(X)):
                grad[i]+= (y_pred[j] - Y[j])*X[j][i]
            W[i] = W[i] - (lr*grad[i])

        if np.sum(grad) == 0 or itr_num > max_itr:
            converged = True
    return W

def getData():
    X = np.array([[0.0,0.0], 
    [0.0,1.0], 
    [1.0,0.0], 
    [1.0,1.0]])
    Y = np.zeros(len(X))
    for i in range(len(X)):
        Y[i] = OR_gate(X[i][0], X[i][1])

    return X, Y

def main():
    X, Y = getData()
    LR=0.1

    X = np.insert(X, 0, 1.0, axis=1)
    W = trainPerceptronWeights(X, Y, lr=LR)
    print("perceptron weights", W)
if __name__ == "__main__":
    main()