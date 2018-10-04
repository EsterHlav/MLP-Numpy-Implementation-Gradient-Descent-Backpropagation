# ©EsterHlav
# July 27, 2017

# Architecture of code inspired from

import numpy as np
from support import load_MNIST_vector, normalizeDataSet, softmax, oneHot, saveNN, restoreNN

class NeuralNetwork():

    def __init__(self, nn_input_dim, nn_hdim, nn_output_dim, seed=0, learningRate=0.01, regTerm=0.01):
        np.random.seed(seed)
        W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
        b1 = np.zeros((1, nn_hdim))
        W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
        b2 = np.zeros((1, nn_output_dim))
        self.model = {'W1':W1, 'W2':W2, 'b1':b1, 'b2':b2 }
        self.sizeInput = nn_input_dim
        self.sizeHidden = nn_hdim
        self.sizeOutput = nn_output_dim
        self.learningRate = learningRate
        self.regTerm = regTerm
        self.trained = False
        self.ValidError = -1
        self.TestError = -1


    def nbParams(self):
        # total number of parameters = input*hidden (W1) + hidden (b1) + hidden*output (W2) + output (b2)
        return self.sizeHidden*(self.sizeInput+1) + self.sizeOutput*(self.sizeHidden+1)

    def __str__(self):
        return "Input dim: {}\nHiddent dim: {}\nOutput dim: {}\nNumber of parameters: {}\nLearning rate hyperparameter: {}\nRegularization hyperparameter: {}\nTrained: {}\nMinimum Validation error: {}\nMinimum Test error: {}".format(self.sizeInput, self.sizeHidden, self.sizeOutput, self.nbParams(), self.learningRate, self.regTerm ,self.trained, self.ValidError, self.TestError)

    # helper to predict the classification of an element
    def predict(self, x):
        # retrieve parameters
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        # compute hidden layer
        z1 = x.dot(W1) + b1
        a1 = np.tanh(z1)
        #compute output layer
        z2 = a1.dot(W2) + b2
        a2 = softmax(z2)
        # return index of element with highest probability
        return np.argmax(a2, axis=1)

    def computeError(self, data):
        X, Y = data
        # retrieve parameters
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        # compute hidden layer
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        #compute output layer
        z2 = a1.dot(W2) + b2
        a2 = softmax(z2)
        predictions = np.argmax(a2, axis=1)
        return np.sum(predictions!=Y)/len(predictions)


    def oneHot(self, x):
        # if sizeOutput=5 for instance, convert 0 to [1,0,0,0,0] or 2 to [0,0,1,0,0]
        return np.eye(self.sizeOutput)[x.reshape(-1)]

    # helper function to compute loss on complete dataset
    def computeLoss(self, data, verbose=False):
        X, Y = data
        # retrieve parameters
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        # compute hidden layer
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        #compute output layer
        z2 = a1.dot(W2) + b2
        a2 = softmax(z2)
        # find logprob over the examples and then add them up
        # use oneHot to obtain something like [0,0,1,0,0]*[0.1,0.2,0.1,0.2,0.2] since entropy is sum over true classes (y_i*log(y_i)) for y_i=1 only if i is true label

        # print("a2: %s" % np.sum(a2, axis=1))
        # print("Min a2: %s" % np.min(a2))
        # print("Min x before log(x): %s" % np.min(np.sum(self.oneHot(Y)*a2, axis=1)))

        crossEnt = -np.log(np.sum(self.oneHot(Y)*a2, axis=1)+ 1e-12) # to take the sum over the classes (avoid log(0))
        #crossEnt = -np.mean(self.oneHot(Y)*np.log(a2+1e-12))

        if verbose:
            print("crossEnt: %s" % crossEnt)

        dataLoss = np.sum(crossEnt)
        # add regulatization term to loss (optional but good to avoid overfitting)
        dataLoss += self.regTerm/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1./X.shape[0] * dataLoss

    def train(self, train, valid, test, nbEpochs=4, decay=0.001, printLoss=True, verbose=False, graph=False, everyEpoch=5):
        X, Y = train
        # retrieve parameters
        W1, b1, W2, b2 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2']
        if graph:
            # to store the losses and errors
            losses = []
            accuracies = []
            errors = []
        # loop over nb of epochs (1 epoch corresponds to update weight with all the data in training set)
        for i in range(nbEpochs):
            print("--> Epoch n°{} running...".format(i))
            # Forward propagation
            z1 = X.dot(W1) + b1
            a1 = np.tanh(z1)
            z2 = a1.dot(W2) + b2
            a2 = softmax(z2)

            # Backpropagation
            delta3 = a2 - self.oneHot(Y)
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(X.T, delta2)
            db1 = np.sum(delta2, axis=0)

            if verbose:
                print("a2: %s" % a2)
                print("Gradient:")
                print("dW2: %s" % dW2)
                print("dW2: %s" % dW1)
                print("delta2: %s" % delta2)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += self.regTerm * W2
            dW1 += self.regTerm * W1

            # learning rate decay to help improve NN
            self.learningRate = self.learningRate * 1/(1 + decay * i)

            # Gradient descent parameter update
            W1 += -self.learningRate * dW1
            b1 += -self.learningRate * db1
            W2 += -self.learningRate * dW2
            b2 += -self.learningRate * db2

            if verbose:
                print("Update parameters:")
                for k in self.model.keys():
                    print("{}: {}".format(k, self.model[k]))

            # Assign new parameters to the model
            self.model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if printLoss and i % everyEpoch == 0:
                print('-'*40)
                loss = self.computeLoss(train)
                acc = self.computeError(train)
                error = self.computeError(valid)
                print ("Loss after iteration {}: {}".format(i, loss ))
                print("Training error after iteration {}: {}%".format(i, error*100))
                print("Validation error after iteration {}: {}%".format(i, error*100))
                print('-'*40)
                if graph:
                    losses.append(loss)
                    accuracies.append(acc*100)
                    errors.append(error*100)

        print("Model trained!")
        self.trained = True
        validError = self.computeError(valid)
        testError = self.computeError(test)
        print("Validation error: {}%  |  Test error: {}%".format(validError*100, testError*100))
        if validError < np.abs(self.ValidError): #abs() for ini since ini is -1
            print("New best validation error!")
            self.ValidError = validError
        if testError < np.abs(self.TestError):
            print("New best test error!")
            self.TestError = testError
        if graph:
            # note that nbEpochs has to be a multiple of everyEpoch...
            import matplotlib.pyplot as plt
            x_values = np.linspace(0, nbEpochs, nbEpochs/everyEpoch+1)[0:(int(nbEpochs/everyEpoch))]
            plt.figure(figsize=(12, 4))
            # plot loss evolution
            plt.subplot(1,3,1)
            plt.title('Evolution of loss')
            plt.plot(x_values, losses)
            # plot error evolution
            plt.subplot(1,3,2)
            plt.title('Evolution of training error')
            plt.plot(x_values, accuracies)
            # plot error evolution
            plt.subplot(1,3,3)
            plt.title('Evolution of validation error')
            plt.plot(x_values, errors)
            plt.show()

    def resetTraining(self):
        self.trained = False
        self.ValidError = -1
        self.TestError = -1
