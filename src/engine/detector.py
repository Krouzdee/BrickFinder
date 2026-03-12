import cv2
import numpy as np
from ultralytics import YOLO
from ..utils import LegoStorage
from collections import defaultdict
import pickle


class LegoDetector:
    """
    Основной класс компьютерного зрения
    YOLO находит детали, SIFT распознаёт их по форме
    """

    def __init__(self, yolov8_model_path='models/lego_detector200.pt'):
        """
        Инициализация детектора.

        Args:
            yolov8_model_path: Путь к предобученной модели YOLOv8.
        """
        self.storage = LegoStorage()
        self.detector = YOLO(yolov8_model_path)

        self.trackers = defaultdict(dict)
        self.track_id = 0

        self.sift = cv2.SIFT_create(
            nfeatures=0,           
            nOctaveLayers=3,
            contrastThreshold=0.04,
            edgeThreshold=10,
            sigma=1.6
        )
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Текущая цель для поиска
        self.current_target_name = ""
        self.current_safe_name = ""
        
        self.target_descriptors = None
        
        self.min_match_count = 15

    def get_sift_features(self, cv2_img):
        """
        Извлекает признаки SIFT из изображения.
        
        Args:
            cv2_img (numpy.ndarray): Изображение в формате BGR.
            
        Returns:
            numpy.ndarray: Дескрипторы SIFT или None
        """
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        _, descriptors = self.sift.detectAndCompute(gray, None)
        
        return descriptors

    def match_descriptors(self, desc1, desc2):
        """
        Сравнивает два набора дескрипторов SIFT.
        
        Args:
            desc1: дескрипторы эталона
            desc2: дескрипторы сцены
            
        Returns:
            float: оценка сходства (0-1)
            int: количество совпадений
        """
        if desc1 is None or desc2 is None:
            return 0.0, 0
        
        if len(desc1) < 2 or len(desc2) < 2:
            return 0.0, 0
        
        try:
            matches = self.flann.knnMatch(desc1, desc2, k=2)
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            match_count = len(good_matches)
            
            if match_count < self.min_match_count:
                return 0.0, match_count
            
            count_score = min(1.0, match_count / 30)
            
            avg_distance = sum(m.distance for m in good_matches) / match_count
            distance_score = max(0, 1 - (avg_distance / 300))
            
            score = (count_score + distance_score) / 2
            
            return score, match_count
            
        except cv2.error:
            return 0.0, 0

    def add_new_target(self, frame, display_name):
        """
        Добавляет новую деталь в базу.
        Сохраняет SIFT признаки.

        Args:
            frame (numpy.ndarray): Кадр с деталью.
            display_name (str): Имя детали.

        Returns:
            str: Путь к сохраненному изображению или False
        """
        res = self.detector(frame, conf=0.4, verbose=False)
        if len(res) > 0 and len(res[0].boxes) > 0:
            b = res[0].boxes.xyxy[0].cpu().numpy().astype(int)
            crop = frame[b[1]:b[3], b[0]:b[2]]
        else:
            crop = frame

        descriptors = self.get_sift_features(crop)
        
        if descriptors is None:
            print("На изображении не найдено достаточно признаков")
            return False
        
        img_path = self.storage.save_reference(
            display_name, 
            crop, 
            descriptors
        )
        return img_path

    def switch_target(self, safe_name):
        """
        Переключает детектор на поиск новой детали из базы.
        Загружает SIFT признаки детали.

        Args:
            safe_name (str): Безопасное имя детали.

        Returns:
            bool: True, если деталь найдена и загружена.
        """
        data = self.storage.load_reference(safe_name)

        if data:
            self.target_descriptors = data['descriptors']
            self.current_target_name = data['name']
            self.current_safe_name = safe_name
            return True
        return False

    def delete_target(self, safe_name):
        """
        Удаляет деталь и сбрасывает текущую цель.

        Args:
            safe_name (str): Безопасное имя детали.

        Returns:
            bool: True если деталь удалилась
        """
        success = self.storage.delete_reference(safe_name)

        if success:
            if safe_name == self.current_safe_name:
                self.target_descriptors = None
                self.current_target_name = ""
                self.current_safe_name = ""
        return success

    def process_frame(self, frame: np.ndarray, threshold_percent: int = 70) -> np.ndarray:
        """
        Главный метод обработки кадра.
        1. YOLO находит все потенциальные объекты.
        2. SIFT сравнивает их с целевой деталью.

        Args:
            frame (numpy.ndarray): Исходный кадр с камеры.
            threshold_percent (int): Порог уверенности в процентах (0-100).

        Returns:
            numpy.ndarray: Кадр с нарисованными рамками.
        """
        if self.target_descriptors is None:
            return frame
        
        threshold = threshold_percent / 100.0

        results = self.detector.track(frame, conf=0.25, persist=True, verbose=False)

        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                descriptors = self.get_sift_features(roi)
                
                sift_score, match_count = self.match_descriptors(
                    self.target_descriptors, 
                    descriptors
                )
                
                if match_count < self.min_match_count:
                    continue

                if sift_score >= threshold:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    score_text = f"{self.current_target_name} {int(sift_score * 100)}%"

                    text_width = len(score_text) * 10
                    cv2.rectangle(
                        frame, 
                        (x1, y1 - 25), 
                        (x1 + text_width, y1), 
                        (0, 255, 0), 
                        -1
                    )

                    cv2.putText(
                        frame, 
                        score_text, 
                        (x1 + 5, y1 - 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        1
                    )

        return frame