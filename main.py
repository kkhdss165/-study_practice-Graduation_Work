from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import tkinter.ttk as ttk
import hair_detech
import facemesh
import facemesh_ver0919

class screen:
    def __init__(self):
        self.root = Tk()
        self.root.title('controller')
        self.my_btn1 = Button(self.root, text='3D Face', command=self.open3d).pack()
        self.my_btn2 = Button(self.root, text='Hair Detect', command=self.openHair).pack()
        self.label = ttk.Label(self.root,image=None)
        self.label.pack()
        self.root.mainloop()

    def open3d(self):
        global my_image
        self.root.filename = filedialog.askopenfilename(initialdir='', title='파일선택', filetypes=(
            ('png files', '*.png'), ('jpg files', '*.jpg'), ('all files', '*.*')))

        my_image = ImageTk.PhotoImage(Image.open(self.root.filename))
        self.label.config(text=self.root.filename,image=my_image)

        # facemesh.createFaceMesh(self.root.filename)
        facemesh_ver0919.createFaceMesh(self.root.filename)

    def openHair(self):
        global my_image
        self.root.filename = filedialog.askopenfilename(initialdir='', title='파일선택', filetypes=(
            ('png files', '*.png'), ('jpg files', '*.jpg'), ('all files', '*.*')))

        my_image = ImageTk.PhotoImage(Image.open(self.root.filename))
        self.label.config(image=my_image)

        hair_detech.runSegement(self.root.filename)

if __name__ == '__main__':
    screen()