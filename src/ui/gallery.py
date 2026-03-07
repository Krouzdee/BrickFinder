import customtkinter as ctk
from typing import Optional, Callable


class ItemCard(ctk.CTkFrame):
    def __init__(self, master, name: str = "1 ИМЯ", image: Optional[ctk.CTkImage] = None,
                 on_white_click: Optional[Callable] = None,
                 on_green_click: Optional[Callable] = None,
                 on_red_click: Optional[Callable] = None, **kwargs):

        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.configure(corner_radius=10, border_width=2)

        self.name_label = ctk.CTkLabel(self, text=name, font=ctk.CTkFont(size=14, weight="bold"))
        self.name_label.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="w")

        self.png_frame = ctk.CTkFrame(self, corner_radius=6, border_width=1, height=50, width=89)
        self.png_frame.grid(row=1, column=0)
        self.png_frame.grid_propagate(False)

        self.png_label = ctk.CTkLabel(self.png_frame, text="")
        self.png_label.pack(expand=True, fill="both")

        if image:
            image.configure(size=(89, 50))
            self.png_label.configure(image=image)
        else:
            self.png_label.configure(text="png")

        self.buttons_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.buttons_frame.grid(row=2, column=0, padx=12, pady=(6, 12), sticky="ew")
        self.buttons_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        self.white_btn = ctk.CTkButton(
            self.buttons_frame, text="", width=34, height=20,
            fg_color="white", hover_color="#E6E6E6", command=on_white_click
        )
        self.white_btn.grid(row=0, column=0, padx=6)
        
        self.green_btn = ctk.CTkButton(
            self.buttons_frame, text="", width=34, height=20,
            fg_color="#00CC00", hover_color="#00AA00", command=on_green_click
        )
        self.green_btn.grid(row=0, column=1, padx=6)
        
        self.red_btn = ctk.CTkButton(
            self.buttons_frame, text="", width=34, height=20,
            fg_color="#FF3333", hover_color="#CC2222", command=on_red_click
        )
        self.red_btn.grid(row=0, column=2, padx=6)


class GalleryWidget(ctk.CTkScrollableFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure((0, 1, 2), weight=1, uniform="col")
        self.items: list[ItemCard] = []
        self.next_num = 1

    def add_item(self, name: Optional[str] = None, image: Optional[ctk.CTkImage] = None, on_white_click: Optional[Callable] = None, on_green_click: Optional[Callable] = None, on_red_click: Optional[Callable] = None) -> ItemCard:
        if name is None:
            name = f"{self.next_num} ИМЯ"
            self.next_num += 1
        card = ItemCard(self, name=name, image=image, on_white_click=on_white_click, on_green_click=on_green_click, on_red_click=on_red_click)
        self.items.append(card)
        self._refresh_layout()
        return card

    def remove_item(self, card: ItemCard) -> None:
        if card in self.items:
            self.items.remove(card)
            card.destroy()
            self._refresh_layout()

    def remove_item_by_index(self, index: int) -> None:
        if 0 <= index < len(self.items):
            card = self.items.pop(index)
            card.destroy()
            self._refresh_layout()

    def clear_all(self) -> None:
        for card in self.items[:]:
            card.destroy()
        self.items.clear()
        self.next_num = 1

    def _refresh_layout(self):
        for widget in self.winfo_children():
            if isinstance(widget, ItemCard):
                widget.grid_forget()
        for i, card in enumerate(self.items):
            row = i // 3
            col = i % 3
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
