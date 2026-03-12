import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from scipy.spatial.distance import cosine
from ..utils import LegoStorage
from PIL import Image
from collections import defaultdict
from typing import Optional, Tuple, Dict, Any, List
from queue import Queue


class LegoDetector:
    """
    Основной класс компьютерного зрения
    """

    def __init__(self, yolov8_model_path: str = 'models/lego_detector200.pt') -> None:
        """
        Инициализация детектора.

        Args:
            yolov8_model_path: Путь к предобученной модели YOLOv8.
        """
        self.storage: LegoStorage = LegoStorage()

        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.detector: YOLO = YOLO(yolov8_model_path)
        if self.device.type == 'cuda':
            self.detector.to('cuda')

        self.trackers: defaultdict = defaultdict(dict)
        self.track_id: int = 0

        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder: torch.nn.Sequential = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        self.encoder.eval()
        self.encoder.to(self.device)

        if self.device.type == 'cuda':
            self.encoder = self.encoder.half()

        self.transform: transforms.Compose = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Текущая цель для поиска
        self.current_target_name: str = ""
        self.current_safe_name: str = ""
        self.target_vector: Optional[np.ndarray] = None
        self.target_color_hist: Optional[np.ndarray] = None

        # Кэш для векторов признаков
        self.vector_cache: Dict[Tuple[int, int, int, int], np.ndarray] = {}
        self.cache_size: int = 50

        # Пул потоков для асинхронной обработки
        self.processing_queue: Queue = Queue(maxsize=5)
        self.result_queue: Queue = Queue()

        self.feature_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.frame_counter: int = 0
        self.recompute_interval: int = 5

    def _clean_cache(self) -> None:
        """Очистка устаревших записей в кэше"""
        if len(self.vector_cache) > self.cache_size:
            items = list(self.vector_cache.items())
            self.vector_cache = dict(items[-self.cache_size // 2:])

    def get_vector(self, cv2_img: np.ndarray) -> np.ndarray:
        """
        Извлекает вектор признаков формы из изображения с помощью ResNet.

        Args:
            cv2_img (numpy.ndarray): Изображение в формате BGR.

        Returns:
            numpy.ndarray: Вектор признаков.

        """
        img_hash = hash(cv2_img.tobytes())
        cache_key = (img_hash, cv2_img.shape[0], cv2_img.shape[1])

        if cache_key in self.vector_cache:
            return self.vector_cache[cache_key].copy()

        img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        img_t = self.transform(img).unsqueeze(0)

        img_t = img_t.to(self.device)
        if self.device.type == 'cuda':
            img_t = img_t.half()

        with torch.no_grad():
            vec = self.encoder(img_t)

        vec_np = vec.flatten().cpu().numpy()

        self.vector_cache[cache_key] = vec_np
        self._clean_cache()

        return vec_np

    def batch_get_vectors(self, rois: List[np.ndarray]) -> List[np.ndarray]:
        """
        Batch extraction of feature vectors for multiple ROIs.
        """
        if not rois:
            return []

        imgs_t = []
        for roi in rois:
            img_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            img_t = self.transform(img)
            imgs_t.append(img_t)

        batch_t = torch.stack(imgs_t).to(self.device)
        if self.device.type == 'cuda':
            batch_t = batch_t.half()

        with torch.no_grad():
            vecs = self.encoder(batch_t)

        vecs_np = [vec.flatten().cpu().numpy() for vec in vecs]
        return vecs_np

    def get_color_hist(self, cv2_img: np.ndarray) -> np.ndarray:
        """
        Извлекает гистограмму только объекта, отсекая темный фон.
        """
        h, w = cv2_img.shape[:2]
        if h * w > 50000:
            scale = np.sqrt(50000 / (h * w))
            new_size = (int(w * scale), int(h * scale))
            cv2_img = cv2.resize(cv2_img, new_size, interpolation=cv2.INTER_LINEAR)

        hsv = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, np.array([0, 50, 40]), np.array([180, 255, 255]))

        hist = cv2.calcHist([hsv], [0, 1], mask, [16, 16], [0, 180, 0, 256])

        cv2.normalize(hist, hist, alpha=1, beta=0, norm_type=cv2.NORM_L1)
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
        h, w = frame.shape[:2]
        if h * w > 640 * 480:
            scale = np.sqrt(640 * 480 / (h * w))
            new_size = (int(w * scale), int(h * scale))
            frame_small = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
        else:
            frame_small = frame

        res = self.detector(frame_small, conf=0.4, verbose=False)
        if len(res) > 0 and len(res[0].boxes) > 0:
            b = res[0].boxes.xyxy[0].cpu().numpy().astype(int)
            if h * w > 640 * 480:
                b = (b / scale).astype(int)
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
            self.vector_cache.clear()
            self.feature_cache.clear()
            return True
        return False

    def delete_target(self, safe_name: str) -> bool:
        """
        Удаляет деталь и сбрасывает текущую цель, если нужно

        Args:
            safe_name (str): Безопасное имя детали.

        Returns:
            bool: True если деталь удалилась
        """
        success = self.storage.delete_reference(safe_name)

        if success:
            if safe_name == self.current_safe_name:
                self.target_vector = None
                self.target_color_hist = None
                self.current_target_name = ""
                self.current_safe_name = ""
                self.vector_cache.clear()
                self.feature_cache.clear()
        return success

    def _process_single_box(self, box: Any, frame: np.ndarray,
                            target_vector: np.ndarray,
                            target_color_hist: np.ndarray,
                            threshold: float, tid: int, cur_vec: np.ndarray, cur_hist: np.ndarray) -> Optional[Dict]:
        """
        Обработка одного бокса (вынесено для возможности распараллеливания)
        """
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if (y2 - y1) < 10 or (x2 - x1) < 10:
            return None

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        shape_sim = 1 - cosine(target_vector, cur_vec)

        color_sim = 1 - cv2.compareHist(target_color_hist, cur_hist, cv2.HISTCMP_BHATTACHARYYA)

        final_sim = (shape_sim * 0.8) + (color_sim * 0.2)

        if final_sim >= threshold:
            return {
                'coords': (x1, y1, x2, y2),
                'score': final_sim,
                'name': self.current_target_name
            }
        return None

    def process_frame(self, frame: np.ndarray, threshold_percent: int = 70) -> np.ndarray:
        """
        Главный метод обработки кадра.

        Args:
            frame (numpy.ndarray): Исходный кадр с камеры.
            threshold_percent (int): Порог уверенности в процентах (0-100).

        Returns:
            numpy.ndarray: Кадр с нарисованными рамками.
        """
        self.frame_counter += 1

        if self.target_vector is None:
            return frame

        threshold = threshold_percent / 100.0

        h, w = frame.shape[:2]
        if h * w > 1280 * 720:
            scale = np.sqrt(1280 * 720 / (h * w))
            new_size = (int(w * scale), int(h * scale))
            frame_small = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
        else:
            frame_small = frame
            scale = 1.0

        results = list(self.detector.track(frame_small, conf=0.25, persist=True,
                                      verbose=False, stream=True))

        detected_boxes: List[Dict] = []

        if results:
            res = results[0]
            if res.boxes:
                boxes = res.boxes
                if boxes.id is not None:
                    tids = boxes.id.cpu().numpy().astype(int)
                else:
                    tids = np.arange(len(boxes))

                target_vector = self.target_vector
                target_color_hist = self.target_color_hist

                new_rois = []
                new_tids = []
                vecs = [None] * len(boxes)
                hists = [None] * len(boxes)

                for i, tid in enumerate(tids):
                    if tid in self.feature_cache and self.frame_counter % self.recompute_interval != 0:
                        vecs[i], hists[i] = self.feature_cache[tid]
                    else:
                        x1, y1, x2, y2 = map(int, boxes[i].xyxy[0])
                        roi = frame_small[y1:y2, x1:x2]
                        new_rois.append(roi)
                        new_tids.append(tid)

                if new_rois:
                    new_vecs = self.batch_get_vectors(new_rois)
                    for i, tid in enumerate(new_tids):
                        vec = new_vecs[i]
                        hist = self.get_color_hist(new_rois[i])
                        self.feature_cache[tid] = (vec, hist)
                        idx = list(tids).index(tid)
                        vecs[idx] = vec
                        hists[idx] = hist

                for i, box in enumerate(boxes):
                    tid = tids[i]
                    result = self._process_single_box(box, frame_small,
                                                      target_vector,
                                                      target_color_hist,
                                                      threshold, tid, vecs[i], hists[i])
                    if result:
                        x1, y1, x2, y2 = result['coords']
                        if scale != 1.0:
                            x1, y1, x2, y2 = [int(c / scale) for c in [x1, y1, x2, y2]]
                        detected_boxes.append({
                            'coords': (x1, y1, x2, y2),
                            'score': result['score'],
                            'name': result['name']
                        })

                current_tids = set(tids)
                self.feature_cache = {k: v for k, v in self.feature_cache.items() if k in current_tids}

        for box_data in detected_boxes:
            x1, y1, x2, y2 = box_data['coords']
            final_sim = box_data['score']

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            score_text = f"{box_data['name']} {int(final_sim * 100)}%"

            (text_width, text_height), _ = cv2.getTextSize(
                score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            y1_text = max(y1 - text_height - 10, 0)

            cv2.rectangle(frame,
                          (x1, y1_text),
                          (x1 + text_width + 10, y1),
                          (0, 255, 0), -1)

            cv2.putText(frame, score_text,
                        (x1 + 5, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

        return frame