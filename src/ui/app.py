import customtkinter as ctk
import cv2
from PIL import Image

class Window(ctk.CTk):
    def __init__(self):
        ctk.CTk.__init__(self)
        self.title("BrickFinder")
        
        self.cap = cv2.VideoCapture(0)

        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.place(relwidth=0.5, relheight=1)

        self.video_label = ctk.CTkLabel(self.input_frame, text="")
        self.video_label.place(relwidth=0.8, relheight=0.5, relx=0.1, rely=0.1)

        self.update_frame()

        self.after(0, lambda: self.state('zoomed'))

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(cv2_image)

            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(860, 600))

            self.video_label.configure(image=ctk_img)
    
        self.after(16, self.update_frame)

