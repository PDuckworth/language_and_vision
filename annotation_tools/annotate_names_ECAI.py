import numpy as np
import cv2
from os import listdir
import os
from os.path import isfile, join
import Tkinter as tk
from PIL import Image, ImageTk
import tkMessageBox
import os.path
import getpass
import sys

class gui():
    """docstring for gui"""
    def __init__(self):
        self.username = getpass.getuser()
        self.dir_saving = '/home/'+self.username+'/Datasets_old/ECAI_dataset_segmented/annotation'
        self.dir1 = '/home/'+self.username+'/Datasets_old/ECAI_dataset_segmented/'
        if not os.path.isdir(self.dir_saving):
            os.mkdir(self.dir_saving)
        self.dir_saving += '/vid'
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
        self.dir2 = self.dir1+'/vid'+str(self.folder)+'/images/'
        self.onlyfiles = sorted([f for f in listdir(self.dir2) if '.jpg' in f])
        self.f1 = 1
        self.check_file()

        #check to see if file exits
    def check_file(self):
        if not os.path.isdir(self.dir_saving+str(self.folder)):
            os.mkdir(self.dir_saving+str(self.folder))
        fname = self.dir_saving+str(self.folder)+'/name.txt'
        if not os.path.isfile(fname):
            self.file_com = open(fname, 'w')
        else:
            print 'warning this person has a file name in its dir, I will append on it.'
            self.file_com = open(fname, 'a')

    def NextVideo(self):
        self.file_com.close()
        self.folder += 1
        if self.folder == 494:
            print 'Finished all videos!'
            sys.exit(1)
        self.dir2 = self.dir1+'/vid'+str(self.folder)+'/images/'
        print 'processing vid num:',self.folder
        self.onlyfiles = sorted([f for f in listdir(self.dir2) if '.jpg' in f])
        self.f1 = 1
        self.check_file()

    def displayText(self):
        """ Display the Entry text value. """
        if self.entryWidget.get().strip() == "":
            tkMessageBox.showerror("Tkinter Entry Widget", "Enter a text value")
        else:
            self.file_com.write(self.entryWidget.get().strip()+'\n')

    def _get_frame(self):
        if self.f1<10:
            return '0000'+str(self.f1)
        elif self.f1<100:
            return '000'+str(self.f1)
        elif self.f1<1000:
            return '00'+str(self.f1)
        elif self.f1<10000:
            return '0'+str(self.f1)

    def show_frame(self):
            if self.f1 >= len(self.onlyfiles):
                self.f1 = 1
            frame = self._get_frame()
            for im_name in self.onlyfiles:
                # print im_name
                if frame in im_name:
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
                    break



def main():
    T = gui()
    T.show_frame()
    T.root.mainloop()

if __name__ == '__main__':
    main()
