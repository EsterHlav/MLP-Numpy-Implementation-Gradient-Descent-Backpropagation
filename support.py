# ©EsterHlav
# July 27, 2017

import numpy as np
import gzip, numpy
import math
import _pickle as Pickle
import matplotlib
import PIL
from PIL import Image, ImageOps, ImageFilter
import scipy
from scipy import ndimage

def load_MNIST_vector():
    # Load the dataset, url: http://yann.lecun.com/exdb/mnist/
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = Pickle.load(f, encoding='latin1')
    f.close()
    return [train_set, valid_set, test_set]

def normalizeDataSet(data):
    X=data[0]
    m = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    mX = np.repeat(m, X.shape[1]).reshape(X.shape)
    stdX = np.repeat(std, X.shape[1]).reshape(X.shape)
    X = (X-mX)/stdX
    newdata = (X,data[1])
    return newdata

# # example to load and test normalization of data:
# train, valid, test = load_MNIST_vector()
# train = normalizeDataSet(train)
# X = train[0]
# print(np.mean(X[2]))
# print(np.std(X[2]))

def normalizeDataPoint(x):
    return (x-np.mean(x))/np.std(x)


def shapeGrid(n):
    width = math.ceil(math.sqrt(n))
    if width*(width-1)>=n:
        return [width,width-1]
    else:
        return [width,width]

# example to check
# for i in range(18):
#     print(i, shapeGrid(i))

def showImages(imgarray, indexes):
    # takes as input a (N*784) set of data and integers (indexes of image to show)
    # and print the corresponding image
    # figure out the size of figure
    n = len(indexes)
    w,l = shapeGrid(n)

    imgarrayX, imgarrayY = imgarray

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(hspace=1, wspace=0.3)
    for i in range(n):
        plt.subplot(w, l, i+1)
        pixels = np.array(imgarrayX[indexes[i]]*255).reshape((28, 28))
        s = "Label: {}".format(imgarrayY[indexes[i]])
        plt.title(s)
        plt.axis('off')
        plt.imshow(pixels, cmap='gray')
    plt.show()

def showPredictedLabels(imgarray, indexes, labels):
    # takes as input a (N*784) set of data, integers (index of images to show) and labels predicted
    # and print the corresponding images as well as the real label and predicted labels

    # figure out the size of figure
    n = len(indexes)
    w,l = shapeGrid(n)

    imgarrayX, imgarrayY = imgarray
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    for i in range(n):
        plt.subplot(w, l, i+1)
        pixels = np.array(imgarrayX[indexes[i]]*255).reshape((28, 28))
        s = "True: {}, Pred: {}".format(imgarrayY[indexes[i]], labels[i])
        plt.title(s)
        plt.axis('off')
        plt.imshow(pixels, cmap='gray')
    plt.show()

# example to try
# idx = [2,9,10,387, 2839, 8473, 10, 89, 87, 1, 12, 26, 28]
# pred = [8, 2, 2, 0, 5, 7, 1, 3, 2, 0, 2, 6, 8]
# showPredictedLabels(valid, idx, pred)


def softmax(x):
    # apply softmax on a vector

    log_c = np.max(x, axis=x.ndim - 1, keepdims=True)
    #for numerical stability
    y = np.sum(np.exp(x - log_c), axis=x.ndim - 1, keepdims=True)
    x = np.exp(x - log_c)/y

    return x

def test_softmax_basic():
    """
    Test softmax (from Stanford assignment 2 CS224D)
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print ("Running basic tests...")
    test1 = softmax(np.array([1,2]))
    print (test1)
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print (test2)
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002]]))
    print (test3)
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

    print ("You should verify these results!\n")

#test_softmax_basic()

def oneHot(x):
    # if sizeOutput=5 for instance, convert 0 to [1,0,0,0,0] or 2 to [0,0,1,0,0]
    return np.eye(10)[x.reshape(-1)]

def saveNN(nn, filename):
    with open(filename+'.pkl', 'wb') as output:
            Pickle.dump(nn, output)
    print('File saved as {}.pkl'.format(filename))

def restoreNN(filename):
    with open(filename+'.pkl', 'rb') as input:
        obj = Pickle.load(input)
    return obj

def getBestShift(img):
    # helper function for preprocessMNIST
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def wrapAffine(img, M, shape):
    # to recreate equivalent of cv2.warpAffine
    res = np.ndarray(shape)
    for x in range(shape[0]):
        for y in range(shape[1]):
            newX = np.inner(M[0,:],np.array([x,y,1]))
            newY = np.inner(M[1,:],np.array([x,y,1]))
            if newX>=0 and newY>=0 and newX<shape[0] and newY<shape[1]:
                res[x,y] = img[int(newX), int(newY)]
            else:
                res[x,y] = 0
    return res


def shift(img,sx,sy):
    # other helper function for preprocessMNIST
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    # shifted = cv2.warpAffine(img,M,(cols,rows))
    # equivalent in scipy
    shifted = wrapAffine(img, M, shape=img.shape)
    return shifted

def preprocessMNIST(img, save=False):
    # PIL to array: x = numpy.asarray(img)
    # array to PIL: im = PIL.Image.fromarray(numpy.uint8(I))

    # 1. From PIL to numpy
    # invert color, resizing and convert to B&W
    img = ImageOps.invert(img)
    img = img.point(lambda p: p > 128 and 255)
    #img = img.filter(ImageFilter.SHARPEN)
    #img = img.resize((300,300)) #, Image.LINEAR)
    img = img.resize((28,28)) #, Image.LINEAR)
    #img = ImageOps.grayscale(img)
    img = img.convert('L')

    if save:
        # to visualize the result in 28x28 before normalization
        img.save("1originalDrawing.png", "png")

    gray = numpy.asarray(img)

    # crop the image
    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:,0]) == 0:
        gray = np.delete(gray,0,1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:,-1]) == 0:
        gray = np.delete(gray,-1,1)

    rows,cols = gray.shape

    # Now we resize our outer box to fit it into a 20x20 box.
    # Let's calculate the resize factor:
    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        # gray = cv2.resize(gray, (cols,rows))
        # equivalent in numpy/PIL
        im = PIL.Image.fromarray(numpy.uint8(gray))
        gray = numpy.asarray(im.resize((cols,rows), PIL.Image.ANTIALIAS))

    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        # gray = cv2.resize(gray, (cols, rows))
        # equivalent in numpy/PIL
        im = PIL.Image.fromarray(numpy.uint8(gray))
        gray = numpy.asarray(im.resize((cols,rows), PIL.Image.ANTIALIAS))

    # But at the end we need a 28x28 pixel image so we add the missing black rows
    # and columns using the np.lib.pad function which adds 0s to the sides.
    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')
    shiftx,shifty = getBestShift(gray)
    shifted = shift(gray,shiftx,shifty)
    gray = shifted
    # save image if required:
    if save:
        img = PIL.Image.fromarray(numpy.uint8(gray))
        img.save("2preprocessedMNIST.png", "png")
    return gray
