from tkinter import *
from tkinter import ttk

"""
def start_up_window():
    _window = Tk()
    _frm = ttk.Frame(_window, padding=10)
    _frm.grid()
    ttk.Label(_window, text="Software Engineering Project - Team3D").grid(column=0, row=0)
    ttk.Button(_window, text="Quit", command=_window.destroy).grid(column=1, row=0)
    _window.mainloop()
"""

root = Tk()
frm = ttk.Frame(root, padding=10)
frm.grid()
ttk.Label(frm, text="Hello World!").grid(column=0, row=0)
ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=0)
root.mainloop()