# ©EsterHlav
# July 27, 2017

import tkinter as tk
from tkinter import *
from MouseDrawing import *
import NN
from support import *
import PIL
from PIL import Image, ImageOps, ImageFilter

class MainApp(tk.Frame):

    def __init__(self, parent):

        # creating a frame for the GUI
        tk.Frame.__init__(self, parent)
        self.parent = parent
        # making the window not resizable by user
        root.resizable(False, False)
        root.overrideredirect(1) # FRAMELESS CANVAS WINDOW

        # function for reseting the canvas button when drawing other number
        def resetAction():
            drawingField.delete("all")

        def quit():
            self.parent.quit()

        # function for printing the prediction made by the NN
        def predictAction():
            # All tricks to save and convert image from this link:
            # https://www.daniweb.com/programming/software-development/threads/363125/saving-a-tkinter-canvas-drawing

            # save the image as ps file and open it with PIL
            drawingField.postscript(file="tmp.eps")
            img = Image.open("tmp.eps")

            # use preprocessMNIST to normalize image like a MNIST image is
            x = preprocessMNIST(img, save=True).flatten()/255

            # normalize the flatten version to make it compatible with NN
            prediction = NN.predict(normalizeDataPoint(x))
            print (str(prediction))
            textToDisplay = "Prediction: "+str(prediction[0])
            w.config(text=textToDisplay)

        # create instance of NN
        NN = restoreNN('savedNN/test150epch')

        # initializing two buttons, one to save the drawing, other one to make the prediction
        button1 = tk.Button(parent, text="quit", command=quit)
        button2 = tk.Button(parent, text="reset", command=resetAction)
        button3 = tk.Button(parent, text="predict", command=predictAction)
        # setting positions of the buttons
        button1.pack(side=TOP, anchor=W)
        button2.pack(side=TOP, anchor=W)
        button3.pack(side=TOP, anchor=W)

        # initializing label that will show us the result of the prediction made by the neural network
        w = Label(parent, text="Prediction: ",bg="lightblue")
        w.pack(side=TOP, anchor=W)

        # canvas field for drawing the digits from 0-9 using mouse
        drawingField = Canvas(self.parent,height=200,width=200)
        center_window(self.parent, 208, 312)
        drawingField.pack()
        # load the function to
        drawingField.bind("<Motion>", movement)
        drawingField.bind("<ButtonPress-1>", buttondown)
        drawingField.bind("<ButtonRelease-1>", buttonup)

root = tk.Tk()
# setting up background color
root.config(background="lightblue")
app = MainApp(root)
app.pack(fill="both", expand=True)
# stopping with exit
root.mainloop()
