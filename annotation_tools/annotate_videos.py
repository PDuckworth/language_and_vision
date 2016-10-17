import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import Tkinter as tk
from PIL import Image, ImageTk
import tkMessageBox
import os.path


class gui():
    """docstring for gui"""
    def __init__(self):
        self.dir1 = '/home/omari/Datasets/scene'
        self.root = tk.Tk()
        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.textFrame = tk.Frame(self.root)
        self.entryLabel = tk.Label(self.textFrame)
        self.entryLabel["text"] = "Enter the text:"
        self.entryLabel.pack(side=tk.LEFT)
        # Create an Entry Widget in textFrame
        self.entryWidget = tk.Entry(self.textFrame)
        self.entryWidget["width"] = 50
        self.entryWidget.pack(side=tk.LEFT)

        self.textFrame.pack()

        self.button1 = tk.Button(self.root, text="Submit", command=self.displayText)
        self.button1.pack()

        self.button2 = tk.Button(self.root, text="Next", command=self.NextVideo)
        self.button2.pack()

        self.lmain = tk.Label(self.root)
        self.lmain.pack()
        self.folder = 1
        self.dir2 = self.dir1+str(self.folder)+'/kinect_rgb/'
        self.onlyfiles = sorted([f for f in listdir(self.dir2) if isfile(join(self.dir2, f))])
        self.f1 = 1
        self.check_file()

        #check to see if file exits
    def check_file(self):
        fname = self.dir1+str(self.folder)+'/commands_test.txt'
        if not os.path.isfile(fname):
            self.file_com = open(fname, 'w')
        else:
            self.file_com = open(fname, 'a')


    def NextVideo(self):
        self.file_com.close()
        self.folder += 1
        print 'folder number:',self.folder
        self.dir2 = self.dir1+str(self.folder)+'/kinect_rgb/'
        self.onlyfiles = sorted([f for f in listdir(self.dir2) if isfile(join(self.dir2, f))])
        self.f1 = 1
        self.check_file()

    def displayText(self):
        """ Display the Entry text value. """
        if self.entryWidget.get().strip() == "":
            tkMessageBox.showerror("Tkinter Entry Widget", "Enter a text value")
        else:
            self.file_com.write(self.entryWidget.get().strip()+'\n')

    def show_frame(self):
            if self.f1 >= len(self.onlyfiles):
                self.f1 = 1
            im_name = self.onlyfiles[self.f1]
            img = cv2.imread(self.dir2+im_name)
            cv2.rectangle(img,(0,0),(250,50),(255,255,255),-1)
            cv2.putText(img,'frame : '+str(self.f1),(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)
            self.f1 += 1
            cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.lmain.imgtk = imgtk
            self.lmain.configure(image=imgtk)
            self.lmain.after(1, self.show_frame)



def main():
    T = gui()
    T.show_frame()
    T.root.mainloop()

if __name__ == '__main__':
    main()
