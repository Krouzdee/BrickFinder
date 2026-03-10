import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from scipy.spatial.distance import cosine
from ..utils import LegoStorage
from PIL import Image
from typing import Optional, Tuple, List, Union

class LegoDetector:
    """
    Основной класс компьютерного зрения
    """
    def __init__(self, yolov8_model_path: str = 'yolov8n.pt', device: Optional[str] = None):
        """
        Инициализация детектора.
        Args:
            yolov8_model_path: Путь к предобученной модели YOLOv8.
            device: Устройство для вычислений ('cuda', 'mps', 'cpu'). Если None, выбирается автоматически.
        """
        self.storage = LegoStorage()

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        print(f"Используется устройство: {self.device}")

        self.detector = YOLO(yolov8_model_path)

        # Загрузка ResNet на выбранное устройство
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.current_target_name: str = ""
        self.current_safe_name: str = ""
        self.target_vector: Optional[np.ndarray] = None
        self.target_color_hist: Optional[np.ndarray] = None

    def get_vector(self, cv2_img: np.ndarray) -> np.ndarray:
        """
        Извлекает вектор признаков формы из изображения с помощью ResNet.

        Args:
            cv2_img (numpy.ndarray): Изображение в формате BGR.

        Returns:
            numpy.ndarray: Вектор признаков.
        
        """
        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        img_t = self.transform(img).unsqueeze(0)
        img_t = img_t.to(self.device)

        with torch.no_grad():
            vec = self.encoder(img_t)
        return vec.cpu().flatten().numpy()


    def get_color_hist(self, cv2_img: np.ndarray) -> np.ndarray:
        """
        Извлекает гистограмму цвета из изображения в пространстве HSV.

        Args:
            cv2_img (numpy.ndarray): Изображение в формате BGR.

        Returns:
            numpy.ndarray: Нормализованная 2D гистограмма (Hue и Saturation).
        """
        hsv = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist


    def add_new_target(self, frame: np.ndarray, display_name: str) -> Optional[str]:
        """
        Добавляет новую деталь в базу на основе загруженного фото.
        Сначала пытается найти объект детектором и вырезать его.

        Args:
            frame (numpy.ndarray): Кадр с деталью.
            display_name (str): Имя детали, которое ввел пользователь.

        Returns:
            bool: True, если деталь успешно добавлена.
        
        """
        res = self.detector(frame, conf=0.4, verbose=False)
        if len(res) > 0 and len(res[0].boxes) > 0:
            b = res[0].boxes.xyxy[0].cpu().numpy().astype(int)
            crop = frame[b[1]:b[3], b[0]:b[2]]
        else:
            crop = frame

        vector = self.get_vector(crop)
        hist = self.get_color_hist(crop)

        img_path = self.storage.save_reference(display_name, crop, vector, hist)
        return img_path

    def switch_target(self, safe_name: str) -> bool:
        """
        Переключает детектор на поиск новой детали из базы.
        Загружает признаки детали из хранилища и сохраняет их в атрибуты класса.

        Args:
            safe_name (str): Безопасное имя детали.

        Returns:
            bool: True, если деталь найдена и загружена.
        """
        data = self.storage.load_reference(safe_name)

        if data:
            self.target_vector = data['vector']
            self.target_color_hist = data['hist']
            self.current_target_name = data['name']
            self.current_safe_name = safe_name
            return True
        return False


    def delete_target(self, safe_name: str) -> bool:
        """
        Удаляет деталь и сбрасывает текущую цель, если нужно
        """
        pass


    def process_frame(self, frame: np.ndarray, threshold_percent: int = 70) -> np.ndarray:
        """
         Главный метод обработки кадра.
        1. Находит все потенциальные объекты на кадре (через YOLO).
        2. Для каждого объекта извлекает признаки формы и цвета.
        3. Сравнивает их с признаками текущей цели.
        4. Рисует рамку, если сходство выше порога.

        Args:
            frame (numpy.ndarray): Исходный кадр с камеры.
            threshold_percent (int): Порог уверенности в процентах (0-100).

        Returns:
            numpy.ndarray: Кадр с нарисованными рамками.
        """
        if self.target_vector is None:
            return frame
        threshold = threshold_percent / 100.0

        results = self.detector.predict(frame, conf=0.25, verbose=False)

        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

            # Сравнение
            cur_vec = self.get_vector(roi)
            shape_sim = 1 - cosine(self.target_vector, cur_vec)

            cur_hist = self.get_color_hist(roi)
            color_sim = cv2.compareHist(self.target_color_hist, cur_hist, cv2.HISTCMP_CORREL)
            color_sim = max(0, color_sim)

            final_sim = (shape_sim * 0.6) + (color_sim * 0.4)

            if final_sim >= threshold:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                score_text = f"{self.current_target_name} {int(final_sim * 100)}%"

                cv2.rectangle(frame, (x1, y1 - 25), (x1 + len(score_text) * 10, y1), (0, 255, 0), -1)

                cv2.putText(frame, score_text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame