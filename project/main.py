from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import hair

def open():
    global my_image  # 함수에서 이미지를 기억하도록 전역변수 선언 (안하면 사진이 안보임)
    root.filename = filedialog.askopenfilename(initialdir='', title='파일선택', filetypes=(
        ('png files', '*.png'), ('jpg files', '*.jpg'), ('all files', '*.*')))

    Label(root, text=root.filename).pack()  # 파일경로 view
    my_image = ImageTk.PhotoImage(Image.open(root.filename))
    Label(image=my_image).pack()  # 사진 view



root = Tk()
root.title('controller')
my_btn1 = Button(root, text='파일열기', command=open).pack()
my_btn2 = Button(root, text='3D Face', command=createFaceMesh).pack()
my_btn3 = Button(root, text='Hair Detect').pack()
root.mainloop()