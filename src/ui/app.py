import customtkinter as ctk
import cv2
from PIL import Image
from CTkScrollableDropdownPP import CTkScrollableDropdown
from pygrabber.dshow_graph import FilterGraph
from tkinter import filedialog
from typing import Optional
from .gallery import GalleryWidget
from ..engine import LegoDetector
from ..utils import LegoStorage

class Window(ctk.CTk):
    def __init__(self) -> None:
        """
        Инициализация главного окна приложения BrickFinder.
        
        Создает основное окно с двумя панелями:
        - Левая панель: видеопоток с камеры и управление распознаванием
        - Правая панель: управление деталями и настройки
        """
        ctk.CTk.__init__(self)
        self.captured_pil: Optional[Image.Image] = None
        self.capture_button: Optional[ctk.CTkButton] = None
        self.save_button: Optional[ctk.CTkButton] = None
        self.upload_button: Optional[ctk.CTkButton] = None
        self.add_preview_label: Optional[ctk.CTkLabel] = None
        self.add_preview_frame: Optional[ctk.CTkFrame] = None
        self.detail_name_entry: Optional[ctk.CTkEntry] = None
        self.add_window: Optional[ctk.CTkToplevel] = None
        self.status: bool = False
        
        self.LegoStorage = LegoStorage()
        self.LegoDetector = LegoDetector()
        
        self.title("BrickFinder")
        self.geometry(f"1200x680+{self.center(1200, 680)}")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.resizable(False, False)

        self.cap = cv2.VideoCapture(0)

        self.input_frame = ctk.CTkFrame(self, fg_color="#333333")
        self.input_frame.place(relwidth=0.6, relheight=1)

        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.place(relwidth=0.4, relheight=1, relx=0.6)

        title_label = ctk.CTkLabel(self.right_frame, text="Управление деталями", font=("Arial", 16, "bold"))
        title_label.place(relx=0.5, rely=0.03, anchor="n")

        self.add_button = ctk.CTkButton(self.right_frame, command=self.open_add_detail, text="Добавить деталь")
        self.add_button.place(relwidth=0.8, relheight=0.05, relx=0.1, rely=0.09)

        self.detail_list = GalleryWidget(self.right_frame)
        self.detail_list.place(relwidth=0.9, relheight=0.35, rely=0.17, relx=0.05)

        settings_label = ctk.CTkLabel(self.right_frame, text="Настройки распознавания", font=("Arial", 16, "bold"))
        settings_label.place(relx=0.5, rely=0.65, anchor="n")

        conf_label = ctk.CTkLabel(self.right_frame, text="Порог уверенности:", font=("Arial", 12))
        conf_label.place(rely=0.7, relx=0.05)

        self.confidence_slider = ctk.CTkSlider(
            self.right_frame,
            from_=0,
            to=100,
            number_of_steps=100,
            command=lambda value: self.confidence_value.configure(text=f"{int(value)}%")
        )
        self.confidence_slider.place(relwidth=0.8, rely=0.75, relx=0.05)
        self.confidence_slider.set(70)

        self.confidence_value = ctk.CTkLabel(self.right_frame, text="70%", font=("Arial", 12))
        self.confidence_value.place(relx=0.87, rely=0.74)

        self.camera_option = ctk.CTkOptionMenu(self.input_frame, width=200)
        self.camera_option.place(relwidth=0.4, relx=0.3, rely=0.7)

        cameras = FilterGraph().get_input_devices()
        self.dropdown = CTkScrollableDropdown(
            self.camera_option,
            height=300,
            width=300,
            pagination=False,
            values=cameras,
            command=self.change_camera
        )
        if not cameras:
            self.camera_option.configure(state="disabled")
            self.camera_option.set("")
        else:
            self.camera_option.set(cameras[0])

        self.video_frame = ctk.CTkFrame(
                self.input_frame,
                fg_color=self.cget("fg_color"),
                corner_radius=0,
                border_color="#1f6aa5",
                border_width=1
        )
        self.video_frame.place(relwidth=0.9, relheight=0.6, relx=0.05, rely=0.05)

        self.video_label = ctk.CTkLabel(self.video_frame, text="", width=646, height=406)
        self.video_label.place()

        self.status_button = ctk.CTkButton(self.input_frame, command=self.change_status, text="Старт")
        self.status_button.place(relwidth=0.2, relheight=0.05, relx=0.4, rely=0.8)

        self.update_frame()

    def open_add_detail(self) -> None:
        """
        Открывает модальное окно для добавления новой детали.
        
        Создает Toplevel окно с полями для ввода названия детали,
        загрузки изображения и кнопками для управления процессом.
        """
        self.add_window = ctk.CTkToplevel(self)
        self.add_window.wm_transient(self)
        self.add_window.grab_set()
        self.add_window.title("Добавить деталь")
        self.add_window.geometry("500x400")
        self.add_window.resizable(False, False)

        ctk.CTkLabel(self.add_window, text="Название детали:", font=("Arial", 12)).place(relx=0.1, rely=0.1)

        self.detail_name_entry = ctk.CTkEntry(self.add_window, placeholder_text="Например: Синий кирпич 2х4")
        self.detail_name_entry.place(relwidth=0.8, relx=0.1, rely=0.2)

        ctk.CTkLabel(self.add_window, text="Изображение детали:", font=("Arial", 12)).place(relx=0.1, rely=0.35)

        self.add_preview_frame = ctk.CTkFrame(self.add_window, width=200, height=150)
        self.add_preview_frame.place(relwidth=0.8, relheight=0.3, relx=0.1, rely=0.45)

        self.add_preview_label = ctk.CTkLabel(self.add_preview_frame, text="Предпросмотр", fg_color="gray")
        self.add_preview_label.place(relwidth=1, relheight=1)

        self.capture_button = ctk.CTkButton(self.add_window, text="Сделать снимок с камеры", command=self.capture_image)
        self.capture_button.place(relwidth=0.35, relx=0.1, rely=0.8)

        self.upload_button = ctk.CTkButton(self.add_window, text="Загрузить фото", command=self.upload_image)
        self.upload_button.place(relwidth=0.35, relx=0.5, rely=0.8)

        self.save_button = ctk.CTkButton(self.add_window, text="Сохранить деталь", fg_color="green", command=self.save_detail)
        self.save_button.place(relwidth=0.5, relx=0.25, rely=0.9)

        def close_popup() -> None:
            self.add_window.destroy()
            self.add_window.grab_release()

        self.add_window.protocol("WM_DELETE_WINDOW", close_popup)

    def change_camera(self, choice: str) -> None:
        """
        Переключает активную камеру на выбранную.
        
        :param choice: Название выбранной камеры из списка доступных
        """
        if self.cap.isOpened():
            self.cap.release()
        self.cap = cv2.VideoCapture(self.dropdown.values.index(choice))
        self.camera_option.set(choice)

    def change_status(self) -> None:
        """
        Переключает статус видеопотока (старт/стоп).
        """
        self.status = not self.status
        if self.status:
            self.video_label.place(x=1, y=1)
            self.status_button.configure(text="Стоп")
        else:
            self.video_label.place_forget()
            self.status_button.configure(text="Старт")

    def update_frame(self) -> None:
        """
        Обновляет кадр видеопотока в реальном времени.
        
        Получает кадр с камеры, обрабатывает его через детектор,
        конвертирует в формат для отображения и обновляет метку видео.
        Вызывается каждые 33мс для плавного отображения.
        """
        if self.cap.isOpened() and self.status:
            ret, frame = self.cap.read()
            frame = self.LegoDetector.process_frame(frame, self.confidence_slider.get())
            if ret:
                cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2_image)

                w, h = self.video_label.winfo_width(), self.video_label.winfo_height()
                if w > 1 and h > 1:
                    ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(w, h))
                    self.video_label.configure(image=ctk_img)

        self.after(33, self.update_frame)

    def capture_image(self) -> None:
        """
        Делает снимок с активной камеры и сохраняет его для добавления детали.
        
        Конвертирует кадр из BGR в RGB формат и создает PIL изображение.
        """
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.captured_pil = Image.fromarray(rgb)
                ctk_img = ctk.CTkImage(light_image=self.captured_pil, dark_image=self.captured_pil, size=(200,113))
                self.add_preview_label.configure(image=ctk_img, text="")

    def upload_image(self) -> None:
        """
        Загружает изображение детали из файловой системы.
        
        Открывает диалог выбора файла и загружает выбранное изображение.
        """
        filename = filedialog.askopenfilename(filetypes=[("Изображение детали лего", "*.jpg *.jpeg *.png *.bmp")])
        if filename:
            self.captured_pil = Image.open(filename)
            ctk_img = ctk.CTkImage(light_image=self.captured_pil, dark_image=self.captured_pil, size=(200,113))
            self.add_preview_label.configure(image=ctk_img, text="")

    def save_detail(self) -> None:
        """
        Сохраняет новую деталь в галерее.
        
        Получает название из поля ввода, создает CTkImage из захваченного
        изображения и добавляет элемент в галерею деталей.
        """
        name = self.detail_name_entry.get().strip()
        if not name or not self.captured_pil:
            return
        if self.captured_pil:
            photo = ctk.CTkImage(dark_image=self.captured_pil)
        else:
            photo = None
        self.detail_list.add_item(name, photo)
        self.add_window.destroy()

    def on_close(self) -> None:
        """
        Обработчик закрытия окна приложения.
        
        Освобождает ресурсы камеры перед завершением работы.
        """
        if self.cap.isOpened():
            self.cap.release()
        self.destroy()
        
    def center(self, x: int, y: int) -> str:
        """
        Вычисляет координаты для центрирования окна на экране.
    
        :param x: Ширина окна.
        :param y: Высота окна.
        :return: Строка геометрии в формате 'X+Y'.
        """
        pos_x = self.winfo_screenwidth() // 2 - x // 2
        pos_y = self.winfo_screenheight() // 2 - y // 2
        return f"{pos_x}+{pos_y}"
