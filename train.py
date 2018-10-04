# ©EsterHlav
# July 27, 2017

from support import *
import NN

# get data
train, valid, test = load_MNIST_vector()

trainNorm = normalizeDataSet(train)
validNorm = normalizeDataSet(valid)
testNorm = normalizeDataSet(test)

# optional: show set of 9 images from dataset
# showImages(train, np.random.randint(len(train[0]), size=9))

# create NN
NN = NN.NeuralNetwork(nn_input_dim=28*28, nn_hdim=500, nn_output_dim=10, seed=1, learningRate=0.005, regTerm=0.015)

# train on 150 epochs showed great results
NN.train(trainNorm, validNorm, testNorm, nbEpochs=150, decay=0.0002, printLoss=True, verbose=False, graph=True, everyEpoch=5)

saveNN(NN, 'savedNN/test150epch')
