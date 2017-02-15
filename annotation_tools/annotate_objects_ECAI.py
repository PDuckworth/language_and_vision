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


class gui():
    """docstring for gui"""
    def __init__(self):
        self.username = getpass.getuser()
        self.dir_saving = '/home/'+self.username+'/Datasets/ECAI_dataset/annotation'
        self.dir1 = '/home/'+self.username+'/Datasets/ECAI_dataset/dataset_images/'
        if not os.path.isdir(self.dir_saving):
            os.mkdir(self.dir_saving)
        self.dir_saving += '/vid'
        self.root = tk.Tk()
        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.read_mouse = 0
        self.finished = 0
        self.mouse_clicks = {2:[],1:[]}
        self.all_objects = {}
        self._HSV_tuples = [(x*1.0/11, 1.0, .5) for x in range(11)]
        self._colors = map(lambda x: colorsys.hsv_to_rgb(*x), self._HSV_tuples)

        self.button1 = tk.Button(self.root, text="Printer_console", command=self.Printer_console)
        self.button1.grid(row=0, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        self.button2 = tk.Button(self.root, text="Printer_paper_tray", command=self.Printer_paper_tray)
        self.button2.grid(row=1, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        self.button3 = tk.Button(self.root, text="Microwave", command=self.Microwave)
        self.button3.grid(row=2, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        self.button4 = tk.Button(self.root, text="Kettle", command=self.Kettle)
        self.button4.grid(row=3, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        self.button5 = tk.Button(self.root, text="Tea_Pot", command=self.Tea_Pot)
        self.button5.grid(row=4, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        self.button6 = tk.Button(self.root, text="Water_Cooler", command=self.Water_Cooler)
        self.button6.grid(row=5, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        self.button7 = tk.Button(self.root, text="Waste_Bin", command=self.Waste_Bin)
        self.button7.grid(row=6, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        self.button8 = tk.Button(self.root, text="Sink", command=self.Sink)
        self.button8.grid(row=7, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        self.button9 = tk.Button(self.root, text="Fridge", command=self.Fridge)
        self.button9.grid(row=8, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        self.button10 = tk.Button(self.root, text="Paper_towel", command=self.Paper_towel)
        self.button10.grid(row=9, column=0, columnspan=1, pady=5, padx=2, sticky="ew")
        self.button11 = tk.Button(self.root, text="Double_doors", command=self.Double_doors)
        self.button11.grid(row=10, column=0, columnspan=1, pady=5, padx=2, sticky="ew")

        self.button12 = tk.Button(self.root, text="next", command=self.NextVideo)
        self.button12.grid(row=11, column=0, columnspan=1, pady=5, padx=2, sticky="ew")

        self.button12a = tk.Button(self.root, text="prev", command=self.PrevVideo)
        self.button12a.grid(row=12, column=0, columnspan=1, pady=5, padx=2, sticky="ew")

        self.button13 = tk.Button(self.root, text="save", command=self.save)
        self.button13.grid(row=13, column=0, columnspan=1, pady=5, padx=2, sticky="ew")

        self.button14 = tk.Button(self.root, text="clear", command=self.clear)
        self.button14.grid(row=14, column=0, columnspan=1, pady=5, padx=2, sticky="ew")

        self.lmain = tk.Label(self.root)
        self.lmain.grid(row=0, column=1, columnspan=1, rowspan=11, pady=10, padx=2, sticky="ew")
        self.lmain.bind("<Button-1>", self.mouse_callback)

        self.folder = 0
        self.NextVideo()
        # self.dir2 = self.dir1+'/vid'+str(self.folder)+'/images/'
        # self.onlyfiles = sorted([f for f in listdir(self.dir2) if '.jpg' in f])
        # self.f1 = 1
        # self.check_file()

    def clear(self):
        self.all_objects = {}

    def save(self):
        """ Display the Entry text value. """
        fname = self.dir_saving+str(self.folder)+'/objects.txt'
        if not os.path.isfile(fname):
            self.file_com = open(fname, 'w')
        else:
            print 'warning this person has an objects file in its dir, I will rewrite it.'
            self.file_com = open(fname, 'w')
        for obj in self.all_objects:
            start = (self.all_objects[obj][0][2][0],self.all_objects[obj][0][2][1],)
            end = (self.all_objects[obj][0][1][0],self.all_objects[obj][0][1][1],)
            self.file_com.write(obj+',')
            self.file_com.write('start(x,y),'+str(start[0])+','+str(start[1])+',')
            self.file_com.write('end(x,y),'+str(end[0])+','+str(end[1])+',')
            self.file_com.write('\n')
        self.file_com.close()

        count = 1
        for im_name in self.onlyfiles:
            img = cv2.imread(self.dir2+im_name)
            cv2.rectangle(img,(0,0),(250,50),(255,255,255),-1)
            cv2.putText(img,'frame : '+str(count),(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)
            img = self.add_objects(img)
            cv2.imwrite(self.dir_saving+str(self.folder)+'/obj_images/'+im_name,img)
            count+=1
        self.clear


    def Printer_console(self):
        self.object = 'Printer_console'; self.read_mouse=2; print 'Printer_console';self.color = self._colors[0]
    def Printer_paper_tray(self):
        self.object = 'Printer_paper_tray'; self.read_mouse=2; print 'Printer_paper_tray';self.color = self._colors[3]
    def Microwave(self):
        self.object = 'Microwave'; self.read_mouse=2; print 'Microwave';self.color = self._colors[1]
    def Kettle(self):
        self.object = 'Kettle'; self.read_mouse=2; print 'Kettle';self.color = self._colors[2]
    def Tea_Pot(self):
        self.object = 'Tea_Pot'; self.read_mouse=2; print 'Tea_Pot';self.color = self._colors[4]
    def Water_Cooler(self):
        self.object = 'Water_Cooler'; self.read_mouse=2; print 'Water_Cooler';self.color = self._colors[5]
    def Waste_Bin(self):
        self.object = 'Waste_Bin'; self.read_mouse=2; print 'Waste_Bin';self.color = self._colors[6]
    def Sink(self):
        self.object = 'Sink'; self.read_mouse=2; print 'Sink';self.color = self._colors[7]
    def Fridge(self):
        self.object = 'Fridge'; self.read_mouse=2; print 'Fridge';self.color = self._colors[8]
    def Paper_towel(self):
        self.object = 'Paper_towel'; self.read_mouse=2; print 'Paper_towel';self.color = self._colors[9]
    def Double_doors(self):
        self.object = 'Double_doors'; self.read_mouse=2; print 'Double_doors';self.color = self._colors[10]

    def mouse_callback(self,event):
        if self.read_mouse:
            self.mouse_clicks[self.read_mouse] = [event.x, event.y]
            print "clicked at", event.x, event.y
            self.read_mouse-=1
            self.finished = 1
        if not self.read_mouse and self.finished:
            print 'created a '+self.object+' object'
            self.all_objects[self.object] = [self.mouse_clicks.copy(),self.color]
            self.finished = 0

        #check to see if file exits
    def check_file(self):
        if not os.path.isdir(self.dir_saving+str(self.folder)):
            os.mkdir(self.dir_saving+str(self.folder))
        if not os.path.isdir(self.dir_saving+str(self.folder)+'/obj_images'):
            os.mkdir(self.dir_saving+str(self.folder)+'/obj_images')

    def PrevVideo(self):
        # self.file_com.close()
        self.folder -= 1
        while(1):
            if self.folder == 0:
                print 'Finished all videos!'
                sys.exit(1)
            fname = self.dir_saving+str(self.folder)+'/objects.txt'
            if not os.path.isfile(fname):
                break
            else:
                self.folder -= 1
        self.dir2 = self.dir1+'/vid'+str(self.folder)+'/images/'
        print 'processing vid num:',self.folder
        self.onlyfiles = sorted([f for f in listdir(self.dir2) if '.jpg' in f])
        self.f1 = 1
        self.check_file()

    def NextVideo(self):
        # self.file_com.close()
        self.folder += 1
        while(1):
            if self.folder == 494:
                print 'Finished all videos!'
                sys.exit(1)
            fname = self.dir_saving+str(self.folder)+'/objects.txt'
            if not os.path.isfile(fname):
                break
            else:
                self.folder += 1
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

    def _get_frame(self,f1):
        if f1<10:
            return '0000'+str(f1)
        elif f1<100:
            return '000'+str(f1)
        elif f1<1000:
            return '00'+str(f1)
        elif f1<10000:
            return '0'+str(f1)

    def add_objects(self,img):
        for obj in self.all_objects:
            # print obj
            font = cv2.FONT_HERSHEY_SIMPLEX
            start = (self.all_objects[obj][0][2][0],self.all_objects[obj][0][2][1],)
            end = (self.all_objects[obj][0][1][0],self.all_objects[obj][0][1][1],)
            color = (self.all_objects[obj][1][0]*255,self.all_objects[obj][1][1]*255,self.all_objects[obj][1][2]*255,)
            cv2.rectangle(img,start,end,color,3)

            start_txt = (self.all_objects[obj][0][2][0]-10,self.all_objects[obj][0][2][1]-15,)
            thik,_ = cv2.getTextSize(obj, font, .8, 1)
            cv2.rectangle(img,(start_txt[0],start_txt[1]-thik[1]),(start_txt[0]+thik[0],start_txt[1]),(235,235,235),-1)
            cv2.putText(img,obj,start_txt,font, .8,color,2)
        return img

    def show_frame(self):
            if self.f1 >= len(self.onlyfiles):
                self.f1 = 1
            frame = self._get_frame(self.f1)
            for im_name in self.onlyfiles:
                # print im_name
                if frame in im_name:
                    img = cv2.imread(self.dir2+im_name)
                    cv2.rectangle(img,(0,0),(250,50),(255,255,255),-1)
                    cv2.putText(img,'frame : '+str(self.f1),(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)
                    img = self.add_objects(img)
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
