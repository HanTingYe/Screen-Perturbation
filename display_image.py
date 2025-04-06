from tkinter import *
from PIL import Image, ImageTk

def display_image():
    top = Tk()  # Importing the tk module
    top.attributes("-fullscreen", True)
    width = top.winfo_screenwidth()
    height = top.winfo_screenheight()
    print(width, height)
    image = Image.open(img_path)
    photo = ImageTk.PhotoImage(image.resize((width, height)))
    label = Label(top)
    label.pack(expand=YES, fill=BOTH)  # Fill the center with an image
    label.configure(image=photo)
    # top.bind("<F11>", top.destroy())
    top.after(top, top.destory)
    top.mainloop()



