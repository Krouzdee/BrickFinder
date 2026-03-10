import customtkinter as ctk
from typing import Optional, Callable, List


class ItemCard(ctk.CTkFrame):
    """Карточка элемента с названием, изображением и тремя кнопками управления."""

    def __init__(
        self,
        master: ctk.CTkBaseClass,
        name: str = "1 ИМЯ",
        image: Optional[ctk.CTkImage] = None,
        on_white_click: Optional[Callable[[int], None]] = None,
        on_green_click: Optional[Callable[[int], None]] = None,
        on_red_click: Optional[Callable[[int], None]] = None,
        **kwargs,
    ) -> None:
        """
        Инициализация карточки элемента.

        Args:
            master: Родительский виджет.
            name: Название элемента (по умолчанию "1 ИМЯ").
            image: Изображение для отображения (CTkImage).
            on_white_click: Функция, вызываемая при нажатии белой кнопки (получает индекс).
            on_green_click: Функция, вызываемая при нажатии зелёной кнопки (получает индекс).
            on_red_click: Функция, вызываемая при нажатии красной кнопки (получает индекс).
            **kwargs: Дополнительные параметры для CTkFrame.
        """
        super().__init__(master, **kwargs)
        self.name = name
        self.grid_columnconfigure(0, weight=1)
        self.configure(corner_radius=10, border_width=2)

        self.name_label = ctk.CTkLabel(
            self,
            text=name,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.name_label.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="w")

        self.png_frame = ctk.CTkFrame(
            self,
            corner_radius=6,
            border_width=1,
            height=50,
            width=89,
        )
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

        self._on_white = on_white_click
        self._on_green = on_green_click
        self._on_red   = on_red_click

        self.white_btn = ctk.CTkButton(
            self.buttons_frame,
            text="",
            width=34,
            height=20,
            fg_color="white",
            hover_color="#E6E6E6",
            command=self._call_white,
        )
        self.white_btn.grid(row=0, column=0, padx=6)

        self.green_btn = ctk.CTkButton(
            self.buttons_frame,
            text="",
            width=34,
            height=20,
            fg_color="#00CC00",
            hover_color="#00AA00",
            command=self._call_green,
        )
        self.green_btn.grid(row=0, column=1, padx=6)

        self.red_btn = ctk.CTkButton(
            self.buttons_frame,
            text="",
            width=34,
            height=20,
            fg_color="#FF3333",
            hover_color="#CC2222",
            command=self._call_red,
        )
        self.red_btn.grid(row=0, column=2, padx=6)


    def set_index(self, index: int) -> None:
        """Устанавливает индекс карточки (вызывается галереей после размещения)."""
        self._index = index

    def _call_white(self) -> None:
        if self._on_white and hasattr(self, "_index"):
            self._on_white(self._index)

    def _call_green(self) -> None:
        if self._on_green and hasattr(self, "_index"):
            self._on_green(self._index)

    def _call_red(self) -> None:
        if self._on_red and hasattr(self, "_index"):
            self._on_red(self._index)


class GalleryWidget(ctk.CTkScrollableFrame):
    """Прокручиваемая галерея с карточками элементов в сетке 3×n."""

    def __init__(self, master: ctk.CTkBaseClass, **kwargs) -> None:
        """
        Инициализация галереи.

        Args:
            master: Родительский виджет.
            **kwargs: Дополнительные параметры для CTkScrollableFrame.
        """
        super().__init__(master, **kwargs)
        self.grid_columnconfigure((0, 1, 2), weight=1, uniform="col")

        self.items: List[ItemCard] = []
        self.next_num: int = 1

    def add_item(
        self,
        name: Optional[str] = None,
        image: Optional[ctk.CTkImage] = None,
        on_white_click: Optional[Callable[[int], None]] = None,
        on_green_click: Optional[Callable[[int], None]] = None,
        on_red_click: Optional[Callable[[int], None]] = None,
    ) -> ItemCard:
        """
        Добавляет новую карточку в галерею.

        Args:
            name: Название элемента (если None — генерируется автоматически).
            image: Изображение для карточки.
            on_white_click: Обработчик белой кнопки (получает индекс карточки).
            on_green_click: Обработчик зелёной кнопки (получает индекс карточки).
            on_red_click: Обработчик красной кнопки (получает индекс карточки).

        Returns:
            ItemCard: Созданная карточка.
        """
        if name is None:
            name = f"{self.next_num} ИМЯ"
            self.next_num += 1

        card = ItemCard(
            self,
            name=name,
            image=image,
            on_white_click=on_white_click,
            on_green_click=on_green_click,
            on_red_click=on_red_click,
        )

        self.items.append(card)
        self._refresh_layout()
        return card

    def remove_item(self, card: ItemCard) -> None:
        """Удаляет указанную карточку из галереи."""
        if card in self.items:
            self.items.remove(card)
            card.destroy()
            self._refresh_layout()

    def remove_item_by_index(self, index: int) -> None:
        """Удаляет карточку по индексу в списке."""
        if 0 <= index < len(self.items):
            card = self.items.pop(index)
            card.destroy()
            self._refresh_layout()

    def clear_all(self) -> None:
        """Удаляет все карточки из галереи и сбрасывает счётчик имён."""
        for card in self.items[:]:
            card.destroy()
        self.items.clear()
        self.next_num = 1

    def _refresh_layout(self) -> None:
        """Перестраивает расположение всех карточек в сетке 3 колонки + обновляет индексы."""
        for widget in self.winfo_children():
            if isinstance(widget, ItemCard):
                widget.grid_forget()

        for i, card in enumerate(self.items):
            row = i // 3
            col = i % 3
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            card.set_index(i)