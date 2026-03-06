import customtkinter as ctk
import cv2
from PIL import Image
from CTkScrollableDropdownPP import CTkScrollableDropdown
from pygrabber.dshow_graph import FilterGraph

class Window(ctk.CTk):
    def __init__(self):
        ctk.CTk.__init__(self)
        self.status = False
        self.title("BrickFinder")
        self.minsize(1000, 562)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cap = cv2.VideoCapture(0)

        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.place(relwidth=0.5, relheight=1)
        
        self.camera_option = ctk.CTkOptionMenu(self.input_frame, width=200)
        self.camera_option.place(relwidth=0.2, relx=0.4, rely=0.66)
        
        cameras = FilterGraph().get_input_devices()
        self.dropdown = CTkScrollableDropdown(
            self.camera_option,
            height=300,
            width=300,
            pagination=False,
            values=cameras,
            command=self.change_camera
        )
        self.camera_option.set(cameras[0])
        
        self.video_frame = ctk.CTkFrame(self.input_frame, fg_color=self.cget("fg_color"), corner_radius=0)
        self.video_frame.place(relwidth=0.9, relheight=0.5, relx=0.05, rely=0.05)
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.place(relwidth=1, relheight=1)

        self.status_button = ctk.CTkButton(self.input_frame, command=self.change_status, text="Старт")
        self.status_button.place(relwidth=0.2, relheight=0.05, relx=0.4, rely=0.6)

        self.update_frame()
        self.after(0, lambda: self.state('zoomed'))

    def change_camera(self, choice):
        if self.cap.isOpened():
            self.cap.release()
        self.cap = cv2.VideoCapture(self.dropdown.values.index(choice))
        self.camera_option.set(choice)

    def change_status(self):
        self.status = not self.status
        if self.status:
            self.video_label.place(relwidth=1, relheight=1)
            self.status_button.configure(text="Стоп")
        else:
            self.video_label.place_forget()
            self.status_button.configure(text="Старт")

    def update_frame(self):
        if self.cap.isOpened()  and self.status:
            ret, frame = self.cap.read()
            if ret:
                cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2_image)

                w, h = self.video_label.winfo_width(), self.video_label.winfo_height()
                if w > 1 and h > 1:
                    ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
                    self.video_label.configure(image=ctk_img)

        self.after(33, self.update_frame)

    def on_close(self):
        if self.cap.isOpened():
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = Window()
    app.mainloop()
