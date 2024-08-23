from tkinter import *
from PIL import Image, ImageTk

def display_image():
    top = Tk()  # 导入tk模块
    top.attributes("-fullscreen", True)
    width = top.winfo_screenwidth()
    height = top.winfo_screenheight()
    print(width, height)
    image = Image.open(
        r'D:\Dropbox\TuD work\ScreenAI_Privacy_Underscreen\UPC_ICCP21_Code-main\optimize_display_POLED_400PPI\miniimagenet\images_test\n0153282900001082.jpg')
    # image = Image.open(img_path)
    photo = ImageTk.PhotoImage(image.resize((width, height)))
    label = Label(top)
    label.pack(expand=YES, fill=BOTH)  # 让图像在中央填充
    label.configure(image=photo)
    # top.bind("<F11>", top.destroy())
    top.after(top, top.destory)
    top.mainloop()



