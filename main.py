# ©EsterHlav
# July 27, 2017

from support import *
import NN

# List of packages used:
# numpy, scipy, PIL, matplotlib, _pickle, tkinter

# get data
train, valid, test = load_MNIST_vector()

testNorm = normalizeDataSet(test)

NN = restoreNN('savedNN/test150epch')

print(NN)

indexes = np.random.randint(0,len(testNorm[0]),20)
predicted = []
for i in indexes:
    label = NN.predict(testNorm[0][i])
    predicted.append(label[0])

showPredictedLabels(test, indexes, predicted)
