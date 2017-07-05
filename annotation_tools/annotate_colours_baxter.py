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
import colorsys
import glob

class gui():
    """docstring for gui"""
    def __init__(self):
        # self.first_click = 1
        # self.all_objects = {}
        self.username = getpass.getuser()
        # self.dir_saving = '/home/'+self.username+'/Datasets/ECAI_dataset/annotation'
        self.dir1 = '/home/'+self.username+'/Datasets/Baxter_Dataset_final/scene'
        # if not os.path.isdir(self.dir_saving):
        #     os.mkdir(self.dir_saving)
        # self.dir_saving += '/vid'
        self.root = tk.Tk()
        self.root.bind('<Escape>', lambda e: self.root.quit())

        self.button1 = tk.Button(self.root, text="red", command=self.red)
        self.button1.grid(row=0, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        self.button2 = tk.Button(self.root, text="green", command=self.green)
        self.button2.grid(row=1, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        self.button2 = tk.Button(self.root, text="blue", command=self.blue)
        self.button2.grid(row=2, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        self.button2 = tk.Button(self.root, text="yellow", command=self.yellow)
        self.button2.grid(row=3, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        self.button2 = tk.Button(self.root, text="black", command=self.black)
        self.button2.grid(row=4, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        self.button2 = tk.Button(self.root, text="white", command=self.white)
        self.button2.grid(row=5, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        self.button2 = tk.Button(self.root, text="brown", command=self.brown)
        self.button2.grid(row=6, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        self.button2 = tk.Button(self.root, text="orange", command=self.orange)
        self.button2.grid(row=7, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        self.button2 = tk.Button(self.root, text="purple", command=self.purple)
        self.button2.grid(row=8, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        self.button2 = tk.Button(self.root, text="pink", command=self.pink)
        self.button2.grid(row=9, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        # self.button2 = tk.Button(self.root, text="duck", command=self.duck)
        # self.button2.grid(row=10, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        # self.button2 = tk.Button(self.root, text="octopus", command=self.octopus)
        # self.button2.grid(row=11, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        # self.button2 = tk.Button(self.root, text="lemon", command=self.lemon)
        # self.button2.grid(row=12, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        # self.button2 = tk.Button(self.root, text="coffee", command=self.coffee)
        # self.button2.grid(row=13, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        # self.button2 = tk.Button(self.root, text="egg", command=self.egg)
        # self.button2.grid(row=14, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        # self.button2 = tk.Button(self.root, text="turtle", command=self.turtle)
        # self.button2.grid(row=15, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        #
        self.lmain = tk.Label(self.root)
        self.lmain.grid(row=0, column=1, columnspan=1, rowspan=16, pady=10, padx=2, sticky="ew")

        self.lmain2 = tk.Label(self.root)
        self.lmain2.grid(row=0, column=2, columnspan=1, rowspan=16, pady=10, padx=2, sticky="ew")

        self.folder = 0
        self.file = 0
        self.NextVideo()

    def _object_selection(self,obj):
        directory = self.dir1+str(self.folder)+"/ground_truth"
        if not os.path.exists(directory):
            os.makedirs(directory)
        f1 = open(self.dir1+str(self.folder)+"/ground_truth/GT_colour_"+str(self.file)+".txt","w")
        f1.write(obj)
        f1.close()
        if self.file==len(self.unique_objects)-1:
            self.NextVideo()
        else:
            self.file+=1

    def red(self):
        self._object_selection("red")

    def green(self):
        self._object_selection("green")

    def blue(self):
        self._object_selection("blue")

    def yellow(self):
        self._object_selection("yellow")

    def black(self):
        self._object_selection("black")

    def white(self):
        self._object_selection("white")

    def brown(self):
        self._object_selection("brown")

    def orange(self):
        self._object_selection("orange")

    def purple(self):
        self._object_selection("purple")

    def pink(self):
        self._object_selection("pink")
    #
    # def octopus(self):
    #     self._object_selection("octopus")
    #
    # def duck(self):
    #     self._object_selection("duck")
    #
    # def lemon(self):
    #     self._object_selection("lemon")
    #
    # def coffee(self):
    #     self._object_selection("coffee")
    #
    # def egg(self):
    #     self._object_selection("egg")
    #
    # def turtle(self):
    #     self._object_selection("turtle")

    def NextVideo(self):
        self.file = 0
        self.folder += 1
        self.unique_pcd = sorted(glob.glob(self.dir1+str(self.folder)+"/clusters/cloud_*.png"))
        self.unique_objects = sorted(glob.glob(self.dir1+str(self.folder)+"/clusters/obj_*.png"))
        self.unique_GT = sorted(glob.glob(self.dir1+str(self.folder)+"/ground_truth/GT_colour_*.txt"))
        print "video: ",self.folder," objects: ",len(self.unique_objects)
        if len(self.unique_objects) == len(self.unique_GT):
            self.NextVideo()

    def show_frame(self):
        img = cv2.imread(self.unique_objects[self.file])
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)
        # self.lmain.after(1, self.show_frame)

        img = cv2.imread(self.unique_pcd[self.file])
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.lmain2.imgtk = imgtk
        self.lmain2.configure(image=imgtk)
        self.lmain2.after(1, self.show_frame)

        # cv2.imshow("img",img)
        # cv2.waitKey(1)

def main():
    T = gui()
    T.show_frame()
    T.root.mainloop()

if __name__ == '__main__':
    main()
