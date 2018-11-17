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
W = np.random.randn(classes, features) * 0.0001

print '==============Program starts here=============='
print 'Xtrain ranges', np.min(Xtrain), np.max(Xtrain)
print 'W ranges', np.min(W), np.max(W)
print 'z ranges ', np.min(W.dot(Xtrain.T)), np.max(W.dot(Xtrain.T))

def hypothesis(X, W):
    ''' X be n * features.
        W be classes * features
    '''
    #z will be classes * n matrix
    z = W.dot(X.T)

    #to ease computation, bring all powers of e in negative
    z -= np.max(z, 0)
    probabilities = np.exp(z)
    probabilities = probabilities / np.sum(probabilities, 0)
    # print 'minimum probabilities ', np.min(probabilities)
    return probabilities


def softmaxLoss(W, Xtrain, Ytrain):
    ''' Let Ytrain be n sized vector with correct labels.
        for each of images
    '''
    hyp = hypothesis(Xtrain, W)#classes * n
    num = hyp[Ytrain, np.arange(Ytrain.__len__())]
    denom = np.sum(hyp, 0)
    return np.sum(-1 * np.log(num / denom)) / hyp.shape[1]

def analyticalGradient(W, Xtrain, Ytrain):
    ''' W => classes * features(number of pixels in image)
        Xtrain => n * features (n images)
        Ytrain => n sized vector, values range from 0 to 9(for 10 classes)
        classes = 10
        features = 3072
        n = 10000
    '''
    print 'Accuracy = ', np.mean( np.argmax(W.dot(Xtrain.T), 0) == Ytrain ) * 100
    gradient = np.zeros(W.shape)
    classes = W.shape[0]
    n = Xtrain.shape[0]
    Z = W.dot(Xtrain.T)#10*n, each column image, each row each class
    Z -= np.max(Z)#To avoid computational overflow
    prob = np.exp(Z) / np.sum(np.exp(Z), 0)
    exclusion = np.zeros((classes, n))
    exclusion[Ytrain,np.arange(n)] = 1
    gradient = (prob - exclusion).dot(Xtrain)
    gradient /= n
    return gradient

def analyticalGradientDescent(W, Xtrain, Ytrain, steps = 10000):
    #W is a random vector of size classes * features
    stepSize = 1e-7
    for i in range(steps):
        print 'Iteration#',i
        W = W - stepSize * analyticalGradient(W, Xtrain, Ytrain)
    return W

#numericalGradient(W, Xtrain, Ytrain)
W = analyticalGradientDescent(W, Xtrain, Ytrain, steps=100)
#W.dump('softmax_reg_trained.dat')

print 'Final Accuracy = ', np.mean( np.argmax(W.dot(Xtrain.T), 0) == Ytrain ) * 100

# numericalGradientDescent(W, Xtrain, Ytrain, 10)
