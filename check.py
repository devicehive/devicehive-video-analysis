from Tkinter import *

master = Tk()

var = IntVar()

v = IntVar()
c = Checkbutton(master, text="Don't show this again", variable=v)
c.var = v
c.pack()

mainloop()