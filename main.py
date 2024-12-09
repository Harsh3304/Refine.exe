import os
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from model import DenoisingCNN, ColourizeCNN  
from scipy.ndimage import median_filter 

file_path = os.getcwd()  

denoise_model = DenoisingCNN()
denoise_model.load_state_dict(torch.load("models/denoising_cnn_150.pth"))
denoise_model.eval()

colourize_model = ColourizeCNN()
colourize_model.load_state_dict(torch.load("models/colorization_model_100eph.pth"))
colourize_model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

def apply_median_filter(image, filter_size=3):
    image_np = np.array(image)
    denoised_image_np = median_filter(image_np, size=filter_size)
    denoised_image = Image.fromarray(denoised_image_np)
    return denoised_image

def np_to_image(img_np):
    img_np = (img_np * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    return ImageTk.PhotoImage(img_pil)

def denoise_image(image):
    image_filtered = apply_median_filter(image, filter_size=3)

    image_tensor = transform(image_filtered).unsqueeze(0)
    with torch.no_grad():
        denoised = denoise_model(image_tensor)
    denoised = denoised.squeeze().permute(1, 2, 0).cpu().numpy()
    return denoised

def colorize_image(image):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        colorized = colourize_model(image_tensor)
    colorized = colorized.squeeze().permute(1, 2, 0).cpu().numpy()
    return colorized

def denoise_screen():
    window.withdraw()  
    denoise_window = Toplevel(window)
    denoise_window.title("Denoise Image")
    denoise_window.geometry("1280x720")
    denoise_window.configure(background='#18242E')

    def add_image():
        global img
        img_path = filedialog.askopenfilename()
        if img_path:
            img = Image.open(img_path).convert('RGB') 
            img_resized = img.resize((520, 520))
            img_display = ImageTk.PhotoImage(img_resized)
            img_label.config(image=img_display)
            img_label.image = img_display

    def start_denoising():
        if img:
            denoised_img = denoise_image(img)
            img_display = np_to_image(denoised_img)
            denoised_label.config(image=img_display)
            denoised_label.image = img_display

    def download_denoised_image():
        if img:
            file = filedialog.asksaveasfilename(defaultextension='.png')
            if file:
                denoised_img = Image.fromarray((denoise_image(img) * 255).astype(np.uint8))
                denoised_img.save(file)
    
    BG_denoise = PhotoImage(file=os.path.join(file_path, "Dependencies", "BG_denoise.png")) 
    BG_denoise_Label = Label(denoise_window, image=BG_denoise, border=0)
    BG_denoise_Label.place(x=0, y=0)

    img_label = Label(denoise_window, bg="#18242E")
    img_label.place(x=300, y=190, width=406, height=406)

    denoised_label = Label(denoise_window, bg="#18242E")
    denoised_label.place(x=792, y=190, width=406, height=406)

    add_image_btn_image = PhotoImage(file=os.path.join(file_path, "Dependencies", "add_image.png"))
    start_btn_image = PhotoImage(file=os.path.join(file_path, "Dependencies", "start_image.png"))
    download_btn_image = PhotoImage(file=os.path.join(file_path, "Dependencies", "download_image.png"))
    back_btn_image = PhotoImage(file=os.path.join(file_path, "Dependencies", "back_btn.png"))

    add_image_button = Button(denoise_window, image=add_image_btn_image, border=0, bg="#18242E", activebackground="#18242E", command=add_image)
    add_image_button.place(x=70, y=180)

    start_button = Button(denoise_window, image=start_btn_image, border=0, bg="#18242E", activebackground="#18242E", command=start_denoising)
    start_button.place(x=70, y=300)

    download_button = Button(denoise_window, image=download_btn_image, border=0, bg="#18242E", activebackground="#18242E", command=download_denoised_image)
    download_button.place(x=70, y=420)

    back_button = Button(denoise_window, image=back_btn_image, border=0, bg="#18242E", activebackground="#18242E", command=lambda: (denoise_window.destroy(), window.deiconify()))
    back_button.place(x=70, y=540)

    denoise_window.mainloop()

def colourize_screen():
    window.withdraw() 
    colourize_window = Toplevel(window)
    colourize_window.title("Colourize Image")
    colourize_window.geometry("1280x720")
    colourize_window.configure(background='#18242E')

    def add_image():
        global img
        img_path = filedialog.askopenfilename()
        if img_path:
            img = Image.open(img_path).convert('L')  
            img_resized = img.resize((520, 520))
            img_display = ImageTk.PhotoImage(img_resized)
            img_label.config(image=img_display)
            img_label.image = img_display

    def start_colorizing():
        if img:
            colorized_img = colorize_image(img)
            img_display = np_to_image(colorized_img)
            colorized_label.config(image=img_display)
            colorized_label.image = img_display

    def download_colorized_image():
        if img:
            file = filedialog.asksaveasfilename(defaultextension='.png')
            if file:
                colorized_img = Image.fromarray((colorize_image(img) * 255).astype(np.uint8))
                colorized_img.save(file)
                
    BG_colourize = PhotoImage(file=os.path.join(file_path, "Dependencies", "BG_colourize.png")) 
    BG_colourize_Label = Label(colourize_window, image=BG_colourize, border=0)
    BG_colourize_Label.place(x=0, y=0)

    img_label = Label(colourize_window, bg="#18242E")
    img_label.place(x=300, y=190, width=406, height=406)

    colorized_label = Label(colourize_window, bg="#18242E")
    colorized_label.place(x=792, y=190, width=406, height=406)

    # Button setup
    add_image_btn_image = PhotoImage(file=os.path.join(file_path, "Dependencies", "add_image.png"))
    start_btn_image = PhotoImage(file=os.path.join(file_path, "Dependencies", "start_image.png"))
    download_btn_image = PhotoImage(file=os.path.join(file_path, "Dependencies", "download_image.png"))
    back_btn_image = PhotoImage(file=os.path.join(file_path, "Dependencies", "back_btn.png"))

    add_image_button = Button(colourize_window, image=add_image_btn_image, border=0, bg="#18242E", activebackground="#18242E", command=add_image)
    add_image_button.place(x=70, y=180)

    start_button = Button(colourize_window, image=start_btn_image, border=0, bg="#18242E", activebackground="#18242E", command=start_colorizing)
    start_button.place(x=70, y=300)

    download_button = Button(colourize_window, image=download_btn_image, border=0, bg="#18242E", activebackground="#18242E", command=download_colorized_image)
    download_button.place(x=70, y=420)

    back_button = Button(colourize_window, image=back_btn_image, border=0, bg="#18242E", activebackground="#18242E", command=lambda: (colourize_window.destroy(), window.deiconify()))
    back_button.place(x=70, y=540)

    colourize_window.mainloop()

def main():
    global BG_main_image, denoise_btn_image, colourize_btn_image, window
    window = Tk()  
    window.geometry("1280x720")
    window.maxsize(1280, 720)
    window.minsize(1280, 720)
    window.title("Refine.exe")
    window.configure(background="white")

    BG_main_image = PhotoImage(file=os.path.join(file_path, "Dependencies", "BG.png"))  
    denoise_btn_image = PhotoImage(file=os.path.join(file_path, "Dependencies", "denoise_option.png"))
    colourize_btn_image = PhotoImage(file=os.path.join(file_path, "Dependencies", "colourize_option.png"))

    bg_label = Label(window, image=BG_main_image)
    bg_label.place(x=0, y=0)

    denoise_button = Button(window, image=denoise_btn_image, border=0, bg="#D9E4E8", activebackground="#D9E4E8", command=denoise_screen)
    denoise_button.place(x=5, y=250)

    colourize_button = Button(window, image=colourize_btn_image, border=0, bg="#D9E4E8", activebackground="#D9E4E8", command=colourize_screen)
    colourize_button.place(x=5, y=350)

    window.mainloop()

main()
