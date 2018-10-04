# ©EsterHlav
# July 27, 2017

from tkinter import *

button = "up"
xold, yold = None, None

def center_window(root, width=300, height=300):
    # geting width and height screen
    screen_width = root.winfo_screenwidth()
    screen_height =  root.winfo_screenheight()

    # calculate position x and y coordinates
    x = (screen_width/2) - (width/2)
    y = (screen_height/2) - (height/2)
    root.geometry('%dx%d+%d+%d' % (width, height, x, y))

# function for when button down we draw
def buttondown(event):
    global button
    button = "down"

# function for when button up we don't draw
def buttonup(event):
    global button, xold, yold
    button = "up"
    # resetting the line when mouse up
    xold = None
    yold = None

# taking event - mouse movement, when mouse moving, draw
def movement(event):
    if button == "down":
        # global variables for all functions
        global xold, yold
        if xold is not None and yold is not None:
            event.widget.create_line(xold,yold,event.x,event.y,smooth=True,width=15)
            event.widget.create_line(xold,yold,event.x,event.y,smooth=True,width=15)
            event.widget.create_line(xold,yold,event.x,event.y,smooth=True,width=15)
        xold = event.x
        yold = event.y


if __name__ == "__main__":
    main()
