import os
import numpy as np

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

data = unpickle(os.getcwd() + '/cifar-10-batches-py/data_batch_1')
Xtrain = data['data']
Ytrain = data['labels']
classes = 10
features = Xtrain.shape[1]
W = np.random.randn(classes, features) * 0.01

print '==============Program starts here=============='
print 'Xtrain ranges', np.min(Xtrain), np.max(Xtrain)
print 'W ranges', np.min(W), np.max(W)
print 'z ranges ', np.min(W.dot(Xtrain.T)), np.max(W.dot(Xtrain.T))

def maximum(a, b):
    if a.shape == b.shape:
        c = np.zeros(a.shape, dtype=a.dtype)
        it = np.nditer(a, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            c[idx] = max(a[idx], b[idx])
            it.iternext()
        return c
    else:
        return -1


def hypothesis(X, W):
    ''' X be n * features.
        W be classes * features
    '''
    #z will be classes * n matrix
    z = W.dot(X.T)

    #to ease computation, bring all powers of e in negative
    z -= np.max(z, 0)
    z = maximum(z, np.full(z.shape, -500.0))
    # print 'minimum z is ', np.min(z), 'maximum z is ', np.max(z)
    probabilities = np.exp(z)
    probabilities = probabilities / np.sum(probabilities, 0)
    # print 'minimum probabilities ', np.min(probabilities)
    return probabilities


def softmaxLoss(W, Xtrain, Ytrain):
    ''' Let Ytrain be n sized vector with correct labels.
        for each of images
    '''
    hyp = hypothesis(Xtrain, W)
    # print 'hypothesis = ', hyp.shape, " => ", hyp[:,0], '\ndenom for this example should be ', np.sum(hyp[:,0])
    num = hyp[Ytrain, np.arange(Ytrain.__len__())]
    denom = np.sum(hyp, 0)

    # print 'min of num is ', min(num), ' and max is ',max(num)

    # print num[1:10], denom[1:10]
    return np.sum(-1 * np.log(num / denom))

def numericalGradient(W, Xtrain, Ytrain):
    h = 0.000001
    f = softmaxLoss(W, Xtrain, Ytrain)
    print "Loss function value = ", f
    gradient = np.zeros(W.shape)
    it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        iw = it.multi_index
        oldValue = W[iw]
        W[iw] = oldValue + h
        fh = softmaxLoss(W, Xtrain, Ytrain)
        W[iw] = oldValue
        gradient[iw] = (fh - f) / h
        it.iternext()
    print 'numerical gradient =>', gradient[:,0]
    return gradient

def numericalGradientDescent(W, Xtrain, Ytrain, steps = 10000):
    #W is a random vector of size classes * features
    stepSize = 1e-300
    for i in range(steps):
        W = W - stepSize * numericalGradient(W, Xtrain, Ytrain)
    return W

def analyticalGradient(W, Xtrain, Ytrain):
    ''' W => classes * features(number of pixels in image)
        Xtrain => n * features (n images)
        Ytrain => n sized vector, values range from 0 to 9(for 10 classes)
        classes = 10
        features = 3072
        n = 10000
    '''
    gradient = np.zeros(W.shape)
    it = np.nditer(W, flags=['multi_index'], op_flags=['readwrite'])
    #print 'gradient size = ', W.shape
    Z = W.dot(Xtrain.T)
    Z -= np.max(Z)#10*n, each column image, each row each class
    print 'Z ranges ', np.min(Z), np.max(Z)
    print 'W ranges ', np.min(W), np.max(W)
    denom = np.sum(np.exp(Z), 0)
    while not it.finished:
        iw = it.multi_index
        x = iw[0]; z = iw[1]
        num = np.exp(Z[x,:].T)
        gr = Xtrain[:, z] * ( ( num / denom) - 1 * (Ytrain == x) )
        gradient[iw] = np.sum( gr )
        it.iternext()
    print 'analytical gradient =>', gradient[:,0]
    return gradient

def analyticalGradientDescent(W, Xtrain, Ytrain, steps = 10000):
    #W is a random vector of size classes * features
    stepSize = 0.0000001
    for i in range(steps):
        W = W - stepSize * analyticalGradient(W, Xtrain, Ytrain)
        print 'loss value =>', softmaxLoss(W, Xtrain, Ytrain)
    return W

#numericalGradient(W, Xtrain, Ytrain)
analyticalGradientDescent(W, Xtrain, Ytrain, 10)

# numericalGradientDescent(W, Xtrain, Ytrain, 10)
